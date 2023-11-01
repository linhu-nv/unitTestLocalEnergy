// #include "stdio.h"
// #include "stdint.h"
// #include "malloc.h"
// #include "string.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <nvToolsExt.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "hashTable.cuh"

#ifdef KV_MAP_OPT
#include "kv_map.cuh"
#endif

#include "timer.cpp"

// alias data type
typedef int8_t int8;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;

// Precision control
#ifdef BIT_ARRAY_OPT
typedef uint32_t dtype;           // hamitonian indices / states
#else
typedef int8 dtype;           // hamitonian indices / states
#endif
typedef float64 coeff_dtype;  // pauli term coeff
const int32 MAX_NQUBITS = 64; // max qubits
const int64 id_stride = 64;   // for BigInt id, 64 qubit using uint64 represent



// Global persistent data
static int32 g_n_qubits = -1;
static int64 g_NK = -1;
static int64 *g_idxs = NULL;
// static float64 *g_coeffs = NULL;
static coeff_dtype *g_coeffs = NULL;
static dtype *g_pauli_mat12 = NULL;
static dtype *g_pauli_mat23 = NULL;

#define ID_WIDTH 2 // for BigInt id

// Float point absolute
#define FABS(x) (((x) < 0.) ? (-(x)) : (x))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Export C interface for julia
extern "C" {
    void set_indices_ham_int_opt(
        const int32 n_qubits,
        const int64 K,
        const int64 NK,
        const int64 *idxs,
        const coeff_dtype *coeffs,
        #ifdef BIT_ARRAY_OPT
        const int8 *pauli_mat12,
        const int8 *pauli_mat23);
        #else
        const dtype *pauli_mat12,
        const dtype *pauli_mat23);
        #endif

    void calculate_local_energy(
        const int64 batch_size,
        const int64 *_states,
        const int64 ist, // assume [ist, ied) and ist start from 0
        const int64 ied,
        const int64 *k_idxs,
        // const int64 *ks,
        const uint64 *ks,
        const psi_dtype *vs,
        const int64 rank,
        const float64 eps,
        psi_dtype *res_eloc_batch);

    void calculate_local_energy_sampling_parallel(
        const int64 all_batch_size,
        const int64 batch_size,
        const int64 *_states,
        const int64 ist,
        const int64 ied,
        const int64 ks_disp_idx,
        const uint64 *ks,
        const psi_dtype *vs,
        const int64 rank,
        const float64 eps,
        psi_dtype *res_eloc_batch);

    void calculate_local_energy_sampling_parallel_bigInt(
        const int64 all_batch_size,
        const int64 batch_size,
        const int64 *_states,
        const int64 ist,
        const int64 ied,
        const int64 ks_disp_idx,
        const uint64 *ks,
        const int64 id_width,
        const psi_dtype *vs,
        const int64 rank,
        const float64 eps,
        psi_dtype *res_eloc_batch);

    void set_gpu_to_mpi(int rank);
    void get_gpu_id(int rank, int print_verbose);
}

void _state2id_huge(const dtype *state, const int64 N, const int64 id_width, const int64 stride, const uint64 *tbl_pow2, uint64 *res_id) {
    memset(res_id, 0, sizeof(uint64) * id_width);
    int max_len = N / stride + (N % stride != 0);
    // int max_len = N / stride;
    // printf("c max_len: %d\n", max_len);

    for (int i = 0; i < max_len; i++) {
        int st = i*stride, ed = MIN((i+1)*stride, N);
        uint64 id = 0;
        for (int j = st, k=0; j < ed; j++, k++) {
            id += tbl_pow2[k] * state[j];
        }
        // printf("c id: %lu\n", id);
        res_id[i] = id;
    }
}

__device__ void _state2id_huge_fuse(const dtype *state_ii, const dtype *pauli_mat12, const int64 N, const int64 id_width, const int64 stride, const uint64 *tbl_pow2, uint64 *res_id) {
    memset(res_id, 0, sizeof(uint64) * id_width);
    int max_len = N / stride + (N % stride != 0);
    // int max_len = N / stride;
    // printf("c max_len: %d\n", max_len);
    for (int i = 0; i < max_len; i++) {
        int st = i*stride, ed = MIN((i+1)*stride, N);
        uint64 id = 0;
        for (int j = st, k=0; j < ed; j++, k++) {
            id += (state_ii[j] ^ pauli_mat12[j])*tbl_pow2[k];
            // id += tbl_pow2[k] * state[j];
        }
        res_id[i] = id;
    }
}

__device__ int _compare_id(const uint64 *s1, const uint64 *s2, const int64 len) {
    for (int i = len-1; i >= 0; i--) {
        if (s1[i] > s2[i]) {
            return 1;
        } else if (s1[i] < s2[i]) {
            return -1;
        }
    }
    return 0;
}

// binary find id among the sampled samples
// idx = binary_find(ks, big_id), [ist, ied) start from 0
// ret_res = 0: find the big_id, and save result in psi_real and psi_imag
__device__ void binary_find_bigInt(const int32 ist, const int32 ied, const uint64 *ks, const psi_dtype *vs, int64 id_width, uint64 *big_id, psi_dtype *psi_real, psi_dtype *psi_imag, int32 *ret_res) {
    int32 _ist = ist, _ied = ied;
    int32 _imd = 0, res = 0xffff;
    while (_ist < _ied) {
        _imd = (_ist + _ied) / 2;
        res = _compare_id(&ks[_imd*id_width], big_id, id_width);
        if (res == 0) {
            // e_loc += coef * vs[_imid]
            *psi_real = vs[_imd * 2];
            *psi_imag = vs[_imd * 2 + 1];
            break;
        }

        if (res == -1) {
            _ist = _imd + 1;
        } else {
            _ied = _imd;
        }
    }
    *ret_res = res;
}


// one MPI <-> one GPU
void set_gpu_to_mpi(int rank) {
    int gpu_cnts = 0;
    cudaGetDeviceCount(&gpu_cnts);
    // cudaGetDevice(&cnt);
    int local_gpu_id = rank % gpu_cnts;
    cudaSetDevice(local_gpu_id);
}

// get bind GPU id of rank
void get_gpu_id(int rank, int print_verbose) {
    int device_id = -1;
    cudaGetDevice(&device_id);
    char pciBusId[256] = {0};
    cudaDeviceGetPCIBusId(pciBusId, 255, device_id);
    if (print_verbose == 1) {
        printf("rank %d bind into local gpu: %d (%s)\n", rank, device_id, pciBusId);
    }
}

// #define BIT_ARRAY_OPT

// convert T_ARR type arr into bit array: arr[len] -> bit_arr[num_uint32]
// assume bit_arr is init zeros
template<typename T_ARR>
void convert2bitarray(const T_ARR *arr, int len, const int num_uint32, uint32_t *bit_arr) {
    // printf("NUM_UINT32: %d\n", num_uint32);
    for(int j = 0; j < num_uint32; ++j) {
        for(size_t i = j*32; i < len && i < (j + 1)*32; ++i) {
            // map 0/-1 -> 0; 1 -> 1
            if (arr[i] == 1) bit_arr[j] |= (arr[i] << (i - j*32));
        }
    }
}

// convert T_ARR type arr into bit array; arr[nrow][ncol] -> bit_arr[nrow][num_uint32]
template<typename T_ARR>
std::pair<int, uint32_t*> convert2bitarray_batch(const T_ARR *arr, int nrow, int ncol) {
    const int num_uint32 = std::ceil(ncol / 32.0);
    const int len_bit_arr = num_uint32 * nrow;
    uint32_t *bit_arr = (uint32_t*)malloc(sizeof(uint32_t) * len_bit_arr);
    memset(bit_arr, 0, sizeof(uint32_t) * len_bit_arr); // init zeros
    for (int i = 0; i < nrow; i++) {
        convert2bitarray(&arr[i*ncol], ncol, num_uint32, &bit_arr[i*num_uint32]);
    }
    return std::make_pair(num_uint32, bit_arr);
}

void print_bit(const uint32_t *bitRepresentation, int len) {
    for(int l = 0; l < len; ++l) {
        auto part = bitRepresentation[l];
        for(int i = 0; i < 32; ++i) {
            printf("%d%s", (part & (1 << i) ? 1 : 0), (i == 31) ? "\n" : ", ");
        }
    }
}

template<typename T>
void print_mat(const T *arr, int nrow, int ncol, std::string mess="") {
    if (mess != "") {
        printf("==%s==\n", mess.c_str());
    }

    for(int i = 0; i < nrow; ++i) {
        for(int j = 0; j < ncol; ++j) {
            printf("%d%s", (arr[i*ncol+j] == 1 ? 1 : 0), (j == ncol-1) ? "\n" : ", ");
        }
    }
}

template<typename T>
void print_mat_bit(const T *arr, int nrow, int ncol, std::string mess="") {
    if (mess != "") {
        printf("==%s==\n", mess.c_str());
    }

    for(int i = 0; i < nrow; ++i) {
        print_bit(&arr[i*ncol], ncol);
    }
}

// Copy Julia data into CPP avoid gc free memory
// ATTENTION: This must called at the first time
void set_indices_ham_int_opt(
    const int32 n_qubits,
    const int64 K,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const int8 *pauli_mat12,
    const int8 *pauli_mat23)
    // const dtype *pauli_mat12,
    // const dtype *pauli_mat23)
{
    g_n_qubits = n_qubits;
    // g_K = K;
    g_NK = NK;

    const size_t size_g_idxs = sizeof(int64) * (NK + 1);
    const size_t size_g_coeffs = sizeof(coeff_dtype) * K;

    // print_mat(pauli_mat12, NK, g_n_qubits, "puali_mat12");

    auto ret1 = convert2bitarray_batch(pauli_mat12, NK, g_n_qubits);
    auto ret2 = convert2bitarray_batch(pauli_mat23, K, g_n_qubits);
    auto num_uint32 = ret1.first;
    auto pauli_mat12_bitarr = ret1.second;
    auto pauli_mat23_bitarr = ret2.second;

    // print_mat_bit(pauli_mat12_bitarr, NK, num_uint32, "puali_mat12_bitarr");
    // print_mat(pauli_mat23, K, g_n_qubits, "puali_mat23");
    // print_mat_bit(pauli_mat23_bitarr, K, num_uint32, "puali_mat23_bitarr");

    const size_t size_g_pauli_mat12 = sizeof(dtype) * (num_uint32 * NK);
    const size_t size_g_pauli_mat23 = sizeof(dtype) * (num_uint32 * K);

    cudaMalloc(&g_idxs, size_g_idxs);
    cudaMalloc(&g_coeffs, size_g_coeffs);
    cudaMalloc(&g_pauli_mat12, size_g_pauli_mat12);
    cudaMalloc(&g_pauli_mat23, size_g_pauli_mat23);
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(g_idxs, idxs, size_g_idxs, cudaMemcpyHostToDevice);
    cudaMemcpy(g_coeffs, coeffs, size_g_coeffs, cudaMemcpyHostToDevice);
    cudaMemcpy(g_pauli_mat12, pauli_mat12_bitarr, size_g_pauli_mat12, cudaMemcpyHostToDevice);
    cudaMemcpy(g_pauli_mat23, pauli_mat23_bitarr, size_g_pauli_mat23, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    float32 real_size = (size_g_idxs + size_g_coeffs + size_g_pauli_mat12 + size_g_pauli_mat23) / 1024.0 / 1024.0;
    printf("----set_indices_ham_int_opt bitarray in CPP_GPU----- real_size = %.4fMB\n", real_size);
}



// find coupled state first.
// if don't exist we just calculate next coupled state and drop the coef calculation
template<int MAXN=64>
__global__ void calculate_local_energy_kernel_bigInt_V1_bitarr(
    const int32 num_uint32,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    const uint64 *ks,
    const int64 id_width,
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch,
    myHashTable ht)
{
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    uint64 big_id[ID_WIDTH];
    //int myfound = 0;
    //int mymissed = 0;

    //float64 clks[4] = {0};
    //clock_t t_st, t_ed;

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;
            // map state -> id
            // int64 j_base = sid * N;
            //int res = 0xffff;
            // int64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            //     id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            // }
            // _state2id_huge_fuse(&state_batch[ii*N], &pauli_mat12[j_base], N, id_width, id_stride, tbl_pow2, big_id);
            //t_st = clock();
            big_id[0] = state_batch[ii*N] ^ pauli_mat12[sid*N];
            if (MAXN >= 64) {
                big_id[0] = ((uint64)(state_batch[ii*N+1] ^ pauli_mat12[sid*N+1]) << 32) | big_id[0];
            }
            if (MAXN >= 96) {
                big_id[1] = state_batch[ii*N+2] ^ pauli_mat12[sid*N+2];
            }
            if (MAXN >= 128) {
                big_id[1] = ((uint64)(state_batch[ii*N+3] ^ pauli_mat12[sid*N+3]) << 32) | big_id[1];
            }
            //t_ed = clock();
            //clks[0] += static_cast<float64>(t_ed - t_st);
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            //int32 _ist = 0, _ied = batch_size;
            //t_st = clock();
            
            //binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            KeyT key(big_id[0], big_id[1]);

            int64_t off = ht.search_key(key);

            if (off != -1) {
                psi_real = ht.values[off].data[0];
                psi_imag = ht.values[off].data[1];
            } else {
                continue;
            }
            //t_ed = clock();
            //clks[1] += static_cast<float64>(t_ed - t_st);
            // printf("index: %d big_id[0]: %llu res: %d\n", index, big_id[0], res);

            coeff_dtype coef = 0.0;

            //t_st = clock();
            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                // int _sum = 0;
                // for (int ik = 0; ik < N; ik++) {
                //     // _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                //     _sum += state_batch[ii*N+ik] & pauli_mat23[i*N+ik];
                // }
                int _sum = __popc(state_batch[ii*N] & pauli_mat23[i*N]);
                if (MAXN >= 64) {
                    _sum += __popc(state_batch[ii*N+1] & pauli_mat23[i*N+1]);
                }
                if (MAXN >= 96) {
                    _sum += __popc(state_batch[ii*N+2] & pauli_mat23[i*N+2]);
                }
                if (MAXN >= 128) {
                    _sum += __popc(state_batch[ii*N+3] & pauli_mat23[i*N+3]);
                }
                // if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                // i_base += N;
            }
            //t_ed = clock();
            //clks[2] += static_cast<float64>(t_ed - t_st);
            // if (FABS(coef) < eps) continue;
            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;

            // printf("ii: %d coef: %f\n", ii, coef);
            // printf("ii=%d e_loc_real=%f\n", ii, e_loc_real);
        }
        //t_st = clock();
        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
        //t_ed = clock();
        //clks[3] += static_cast<float64>(t_ed - t_st);
    }
    
    //float64 _sum = clks[0] + clks[1] + clks[2] + clks[3];
    //if (threadIdx.x == 0)
    //printf("tid: %d clks: %.3lf %.3lf %.3f %.3f\n", index, clks[0]/_sum, clks[1]/_sum, clks[2]/_sum, clks[3]/_sum);
}



void calculate_local_energy_sampling_parallel_bigInt(
    const int64 all_batch_size,
    const int64 batch_size,
    const int64 *_states,
    const int64 ist,
    const int64 ied,
    const int64 ks_disp_idx,
    const uint64 *ks,
    const int64 id_width,
    const psi_dtype *vs,
    const int64 rank,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    printf("BIT_ARRAY_OPT BigInt\n");
    Timer timer[4];
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    timer[0].start();
    auto ret = convert2bitarray_batch(_states, batch_size, g_n_qubits);
    timer[0].stop("convert2bitarray_batch");
    const int num_uint32 = ret.first;
    uint32 *states = ret.second;
    const size_t size_states = sizeof(uint32) * batch_size * num_uint32;

    const size_t size_ks = sizeof(uint64) * all_batch_size * id_width;
    const size_t size_vs = sizeof(psi_dtype) * all_batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64 *d_ks = NULL;
    dtype *d_states = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    
    // printf("rank: %d all_batch_size: %d, batch_size: %d ks_disp_idx: %d batch_size_cur_rank: %d\n", rank, all_batch_size, batch_size, ks_disp_idx, batch_size_cur_rank);
    // if (rank == 0) {
    //     for (int i = 0; i < all_batch_size; i++) {
    //         printf("ks[%ld]=%ld, vs=(%f, %f)\n", i, ks[i], vs[i*2], vs[i*2+1]);
    //     }
    // }
    // puts("\n");

    timer[2].start();
    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
    // calculate_local_energy_kernel_bigInt<<<nblocks, nthreads>>>(

    //TODO:build hash table
    
    float avg2cacheline = 0.3;
    float avg2bsize = 0.55;

    int cacheline_size = 128/sizeof(KeyT);
    int avg_size = cacheline_size*avg2cacheline;
    int bucket_size = avg_size/avg2bsize;
    int bucket_num = (all_batch_size + avg_size - 1)/avg_size;

    myHashTable ht;

    while(!buildHashTable(ht, (KeyT *)d_ks, (ValueT *)d_vs, bucket_num, bucket_size, all_batch_size)) {
        bucket_size = 1.4*bucket_size;
        avg2bsize = (float)avg_size/bucket_size;
        printf("Build hash table failed! The avg2bsize is %f now. Rebuilding... ...\n", avg2bsize);
    }

    /*long *found, *missed;
    cudaMalloc((void **)&found, sizeof(long)*nblocks*4);
    cudaMalloc((void **)&missed, sizeof(long)*nblocks*4);
    cudaMemset(found, 0, sizeof(long)*nblocks*4);
    cudaMemset(found, 0, sizeof(long)*nblocks*4);*/

    //nvtxRangePushA("lck");
    if (n_qubits <= 32) {
        calculate_local_energy_kernel_bigInt_V1_bitarr<32><<<nblocks, nthreads>>>(
            num_uint32,
            g_NK,
            idxs,
            coeffs,
            pauli_mat12,
            pauli_mat23,
            all_batch_size,
            batch_size_cur_rank,
            ks_disp_idx,
            &d_states[ist*N],
            d_ks,
            id_width,
            d_vs,
            eps,
            d_res_eloc_batch,
            ht);
            //found,
            //missed);
    } else if (n_qubits <= 64) {
        calculate_local_energy_kernel_bigInt_V1_bitarr<64><<<nblocks, nthreads>>>(
            num_uint32,
            g_NK,
            idxs,
            coeffs,
            pauli_mat12,
            pauli_mat23,
            all_batch_size,
            batch_size_cur_rank,
            ks_disp_idx,
            &d_states[ist*N],
            d_ks,
            id_width,
            d_vs,
            eps,
            d_res_eloc_batch,
            ht);
            //found,
            //missed);
    } else if (n_qubits <= 96) {
        calculate_local_energy_kernel_bigInt_V1_bitarr<96><<<nblocks, nthreads>>>(
        // calculate_local_energy_kernel_bigInt_bitarr<96><<<nblocks, nthreads>>>(
            num_uint32,
            g_NK,
            idxs,
            coeffs,
            pauli_mat12,
            pauli_mat23,
            all_batch_size,
            batch_size_cur_rank,
            ks_disp_idx,
            &d_states[ist*N],
            d_ks,
            id_width,
            d_vs,
            eps,
            d_res_eloc_batch,
            ht);
            //found,
            //missed);
    } else if (n_qubits <= 128) {
        calculate_local_energy_kernel_bigInt_V1_bitarr<128><<<nblocks, nthreads>>>(
            num_uint32,
            g_NK,
            idxs,
            coeffs,
            pauli_mat12,
            pauli_mat23,
            all_batch_size,
            batch_size_cur_rank,
            ks_disp_idx,
            &d_states[ist*N],
            d_ks,
            id_width,
            d_vs,
            eps,
            d_res_eloc_batch,
            ht);
            //found,
            //missed);
    } else {
        printf("Error: only support n_qubits <= 128\n");
    }
    //nvtxRangePop();
    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    freeHashTable(ht);
    /*long found_total = thrust::reduce(thrust::device, found, found + nblocks*4);
    long missed_total = thrust::reduce(thrust::device, missed, missed + nblocks*4);*/
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[2].stop("bigIntBitarray: local_energy_kernel");
    //printf("total missed is %ld, total found is %ld\n", missed_total, found_total);
    printf("batchsize is %d, g_nk is %d, id_width is %d\n", batch_size_cur_rank, g_NK, id_width);

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
    //cudaFree(found);
    //cudaFree(missed);
}
