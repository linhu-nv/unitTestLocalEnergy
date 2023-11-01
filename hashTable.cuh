#include <stdlib.h>

// typedef float64 psi_dtype;
typedef float psi_dtype;

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)


struct KeyT{
    char data[16];
    __device__ __host__ KeyT() {}
    __device__ __host__  KeyT(int64_t v1) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
        ptr[1] = v1;
    }
    __device__ __host__  KeyT(int64_t v1, int64_t v2) {
        int64_t* ptr = static_cast<int64_t *>((void*)data);
        ptr[0] = v1;
        ptr[1] = v2;
    }
    __device__ __host__ bool operator == (const KeyT key) {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (d1[0] == _d1[0] && d2[0] == _d2[0]) ? true : false;
    }
    __device__ __host__ bool operator < (const KeyT key) const {
        int64_t* d1 = (int64_t *)key.data;
        int64_t* d2 = (int64_t *)(key.data + 8);
        int64_t* _d1 = (int64_t *)data;
        int64_t* _d2 = (int64_t *)(data + 8);
        return (_d1[0] < d1[0]) ||  (_d1[0] == d1[0] && _d2[0] < d2[0]);
    }
    __device__ __host__ void print(int matched) {
	    int* ptr = (int*)data;
	    printf("%d %d %d %d is %d\n", ptr[0], ptr[1], ptr[2], ptr[3], matched);
	    return ;
    }
    __device__ __host__ void set_to_one(int p) {
        data[p/8] |= (1<<(p%8));
    }
    __device__ __host__ bool if_set_to_one(int p) {
        return (data[p/8] >> (p%8)) & 1; 
    }
};
struct ValueT{
    psi_dtype data[2];
};


__device__ __host__ int myHashFunc(KeyT value, int threshold) {
    //BKDR hash
    unsigned int seed = 31;
    char* values = static_cast<char*>(value.data);
    int len = sizeof(KeyT);
    unsigned int hash = 171;
    while(len--) {
        char v = (~values[len-1])*(len&1) + (values[len-1])*(~(len&1));
        hash = hash * seed + (v&0xF);
    }
    return (hash & 0x7FFFFFFF) % threshold;
    //AP hash
    /*unsigned int hash = 0;
    int len = sizeof(KeyT);
    char* values = static_cast<char*>(value.data);
    for (int i = 0; i < len; i++) {
        if ((i & 1) == 0) {
            hash ^= ((hash << 7) ^ (values[i]&0xF) ^ (hash >> 3));
        } else {
            hash ^= (~((hash << 11) ^ (values[i]&0xF) ^ (hash >> 5)));
        }
    }
    return (hash & 0x7FFFFFFF)%threshold;*/
    //return ((value & 0xff)+((value>>8) & 0xff)+((value>>16) &0xff)+((value >> 24)&0xff))%threshold;

}
#define _len 16
__device__ __host__ int hashFunc1(KeyT value, int threshold) {
     
    int p = 16777619;
    int hash = (int)2166136261L;
    //int _len = sizeof(KeyT);
    char *values = static_cast<char*>(value.data);
#pragma unroll
    for (int i = 0; i < _len; i ++)
            hash = (hash ^ values[i]) * p;
    hash += hash << 13;
    hash ^= hash >> 7;
    hash += hash << 3;
    hash ^= hash >> 17;
    hash += hash << 5;
    return (hash & 0x7FFFFFFF) % threshold;
}

__device__ __host__ int hashFunc2(KeyT value, int threshold) {
    /*int len = sizeof(KeyT);
    char *values = static_cast<char*>(value.data);
    int hash = 324223113;
    for (int i = 0; i < len; i ++) 
        hash = (hash<<4)^(hash>>28)^values[i];
    return (hash & 0x7FFFFFFF) % threshold;*/

    unsigned int seed = 12313;
    char* values = static_cast<char*>(value.data);
    //int _len = sizeof(KeyT);
    unsigned int hash = 711371;
#pragma unroll
    for (int i = _len; i > 0; i --) {
        char v = (~values[i-1])*(i&1) + (values[i-1])*(~(i&1));
        hash = hash * seed + (v&0xF);
    }
    return (hash & 0x7FFFFFFF) % threshold;
}

__device__ __host__ int hashFunc3(KeyT value, int threshold) {
    //int _len = sizeof(KeyT);
    char *values = static_cast<char*>(value.data);
    int b    = 378551;
    int a    = 63689;
    int hash = 0;
#pragma unroll
    for(int i = 0; i < _len; i++)
    {
      hash = hash * a + values[i];
      a    = a * b;
    }
    return (hash & 0x7FFFFFFF)%threshold;
    
}


#define BFT uint32_t
struct myHashTable {
    KeyT* keys;
    ValueT* values;
    int* bCount;
    BFT* bf;
    int bNum;
    int bSize;
    __inline__ __device__ int64_t search_key(KeyT key, int& filtered) {
        int hashvalue = myHashFunc(key, bNum);
        int my_bucket_size = bCount[hashvalue];
        KeyT* list = keys + (int64_t)hashvalue*bSize;
        int thre = sizeof(BFT)*8;
        BFT my_bf = bf[hashvalue];
        if (!((my_bf>>hashFunc1(key, thre))&1)
            || !((my_bf>>hashFunc2(key, thre))&1) 
            || !((my_bf>>hashFunc3(key, thre))&1)) 
            {
                filtered ++;
                return -1;
            }
            
        for (int i = 0; i < my_bucket_size; i ++) {
            if (list[i] == key) {
                return hashvalue*bSize + i;
            }
        }
        return -1;
    }
};

__global__ void build_hashtable_kernel(myHashTable ht, KeyT* all_keys, ValueT* all_values, int ele_num, int* build_failure) {
    int bucket_num = ht.bNum;
    int bucket_size = ht.bSize;
    KeyT* keys = ht.keys;
    ValueT* values = ht.values;
    int* bucket_count = ht.bCount;
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int i = thread_idx; i < ele_num; i =  i+total_threads) {
        KeyT my_key = all_keys[i];
        ValueT my_value = all_values[i];
        int hashed_value = myHashFunc(my_key, bucket_num);
        int write_off = atomicAdd(bucket_count + hashed_value, 1);
        if (write_off >= bucket_size) {
            build_failure[0] = 1;
            //printf("keyIdx is %d, hashed value is %d, now size is %d, error\n", i, hashed_value, write_off);
            break;
        }
        keys[hashed_value*bucket_size + write_off] = my_key;
        values[hashed_value*bucket_size + write_off] = my_value;
    }
    return ;
}
__global__ void build_hashtable_bf_kernel(myHashTable ht) {
    int bucket_num = ht.bNum;
    int bucket_size = ht.bSize;
    KeyT* keys = ht.keys;
    int* bucket_count = ht.bCount;
    int thread_idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int bid = thread_idx; bid < bucket_num; bid += gridDim.x * blockDim.x) {
        int my_bsize = bucket_count[bid];
        BFT my_bf = 0;
        for (int e = 0; e < my_bsize; e ++) {
            KeyT my_value = keys[bid * bucket_size + e];
            int hv = hashFunc1(my_value, sizeof(BFT)*8);
            my_bf |= (1<<hv);
            hv = hashFunc2(my_value, sizeof(BFT)*8);
            my_bf |= (1<<hv);
            hv = hashFunc3(my_value, sizeof(BFT)*8);
            my_bf |= (1<<hv);
        }
        ht.bf[bid] = my_bf;
    }
    return ;
}

void freeHashTable(myHashTable ht) {
    CUDA_TRY(cudaFree(ht.keys));
    CUDA_TRY(cudaFree(ht.values));
    CUDA_TRY(cudaFree(ht.bCount));
    CUDA_TRY(cudaFree(ht.bf));
}

bool buildHashTable(myHashTable &ht, KeyT* all_keys, ValueT* all_values, int bucket_num, int bucket_size, int ele_num) {
    

    ht.bNum = bucket_num;
    ht.bSize = bucket_size;

    printf("bnum is %d, bsize is %d, ele num is %d\n", bucket_num, bucket_size, ele_num);

    int total_size = ht.bNum * ht.bSize;
    CUDA_TRY(cudaMalloc((void **)&ht.keys, sizeof(KeyT)*total_size));
    CUDA_TRY(cudaMalloc((void **)&ht.values, sizeof(ValueT)*total_size));
    CUDA_TRY(cudaMalloc((void **)&ht.bCount, sizeof(int)*bucket_num));
    CUDA_TRY(cudaMalloc((void **)&ht.bf, sizeof(BFT)*bucket_num));
    CUDA_TRY(cudaMemset(ht.bCount, 0, sizeof(int)*bucket_num));
    CUDA_TRY(cudaMemset(ht.bf, 0, sizeof(BFT)*bucket_num));
    
    int* build_failure;
    CUDA_TRY(cudaMalloc((void **)&build_failure, sizeof(int)));
    CUDA_TRY(cudaMemset(build_failure, 0, sizeof(int)));

    //build hash table kernel
    //TODO: here we use atomic operations for building hash table for simplicity.
    //If we need better performance for this process, we can use multi-split.

    cudaEvent_t start, stop;
    float esp_time_gpu;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&stop));
    CUDA_TRY(cudaEventRecord(start, 0));

    int block_size = 256;
    int block_num = 2048;
    build_hashtable_kernel<<<block_num, block_size>>>(ht, all_keys, all_values, ele_num, build_failure);
    CUDA_TRY(cudaDeviceSynchronize());
    build_hashtable_bf_kernel<<<block_num, block_size>>>(ht);
    CUDA_TRY(cudaDeviceSynchronize());

    CUDA_TRY(cudaEventRecord(stop, 0));
    CUDA_TRY(cudaEventSynchronize(stop));
    CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
    printf("Time for build_hashtable_kernel is: %f ms\n", esp_time_gpu);

    /*int* h_hash_count = new int[bucket_num];
    cudaMemcpy(h_hash_count, ht.bCount, sizeof(int)*bucket_num, cudaMemcpyDeviceToHost);
    for (int i = 0; i < bucket_num; i ++)
        printf("%d ", h_hash_count[i]);
    printf("\n");
    delete [] h_hash_count;*/

    /*KeyT *h_keys = new KeyT[bucket_num*bucket_size];
    cudaMemcpy(h_keys, ht.keys, sizeof(KeyT)*bucket_size*bucket_num, cudaMemcpyDeviceToHost);
    printf("here is the bucket:\n");
    for (int i = 0; i < bucket_num; i ++) {
        printf("bucket %d:\n", i);
        for (int j = 0; j < h_hash_count[i]; j ++) {
            h_keys[i*bucket_size + j].print(0);
        }
    }
    printf("\n");
    delete [] h_keys;*/
    


    //build success check
    int* build_flag = new int[1];
    CUDA_TRY(cudaMemcpy(build_flag, build_failure, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaDeviceSynchronize());
    bool return_state = build_flag[0] == 0 ? true : false;
    if (build_flag[0] == 1) {
        CUDA_TRY(cudaFree(ht.keys));
        CUDA_TRY(cudaFree(ht.values));
        CUDA_TRY(cudaFree(ht.bCount));
        CUDA_TRY(cudaFree(ht.bf));
    } else {
        printf("build hash table success\n");
    }
    delete [] build_flag;
    CUDA_TRY(cudaFree(build_failure));
    return return_state;
}