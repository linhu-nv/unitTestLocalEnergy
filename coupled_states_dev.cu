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

// typedef float64 psi_dtype;
typedef float32 psi_dtype;

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

// binary find id among the sampled samples
// idx = binary_find(ks, big_id), [ist, ied) start from 0
// ret_res = 0: find the big_id, and save result in psi_real and psi_imag
__device__ void binary_find_bigInt_cond(const int32 ist, const int32 ied, const uint64 *ks, const psi_dtype *vs, int64 id_width, uint64 *big_id, psi_dtype *psi_real, psi_dtype *psi_imag, int32 *ret_res) {
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

        // if (res == -1) {
        //     _ist = _imd + 1;
        // } else {
        //     _ied = _imd;
        // }
        _ist = (res == -1) ? _imd + 1 : _ist;
        _ied = (res == -1) ? _ied : _imd;
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
#ifdef BIT_ARRAY_OPT
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
#endif

/**
 * Calculate local energy by fusing Hxx' and summation.
 * Args:
 *     n_qubits: number of qubit
 *     idxs: index of pauli_mat23 block
 *     coeffs: pauli term coefficients
 *     pauli_mat12: extract info of pauli operator 1 and 2 only for new states calculation
 *     pauli_mat23: extract info of pauli operator 2 and 3 only for new coeffs calculation
 *     batch_size: samples number which ready to Hxx'
 *     state_batch: samples
 *     ks: map samples into id::Int Notion: n_qubits <= 64!
 *     vs: samples -> psis (ks -> vs)
 *     eps: dropout coeff < eps
 * Returns:
 *     res_eloc_batch: save the local energy result with complex value,
 *                     res_eloc_batch(1/2,:) represent real/imag
 * */ 
__global__ void calculate_local_energy_kernel(
    const int32 n_qubits,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    // const int64 *ks,
    const uint64_t *ks,
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    // replace branch to calculate state -> id
    // __shared__ int64 tbl_pow2[MAX_NQUBITS];
    __shared__ uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }
    float64 clks[4] = {0};
    clock_t t_st, t_ed;
    // loop all samples
    // for (int ii = 0; ii < batch_size_cur_rank; ii++) {
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        int64 i_base = 0;
        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef = 0.0;

            t_st = clock();
            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                int _sum = 0;
                for (int ik = 0; ik < N; ik++) {
                    _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                }
                // if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                i_base += N;
            }
            // printf("ii: %d coef: %lf\n", ii, coef);
            t_ed = clock();
            clks[0] += static_cast<float64>(t_ed - t_st);
            // filter value < eps
            if (FABS(coef) < eps) {
                continue;
            }

            t_st = clock();
            // printf("ii: %d coef: %f\n", ii, coef);
            // map state -> id
            int64 j_base = sid * N;
            // int64 id = 0;
            uint64 id = 0;
            for (int ik = 0; ik < N; ik++) {
                id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            }
            t_ed = clock();
            clks[1] += static_cast<float64>(t_ed - t_st);

            t_st = clock();
            #if 0
            // linear find id among the sampled samples
            for (int _ist = 0; _ist < batch_size; _ist++) {
                if (ks[_ist] == id) {
                    e_loc_real += coef * vs[_ist * 2];
                    e_loc_imag += coef * vs[_ist * 2 + 1];
                    break;
                }
            }
            #else
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size, _imd = 0;
            while (_ist < _ied) {
                _imd = (_ist + _ied) / 2;
                if (ks[_imd] == id) {
                    // e_loc += coef * vs[_imid]
                    e_loc_real += coef * vs[_imd * 2];
                    e_loc_imag += coef * vs[_imd * 2 + 1];
                    break;
                }

                if (ks[_imd] < id) {
                    _ist = _imd + 1;
                } else {
                    _ied = _imd;
                }
                // int res = ks[_imd] < id;
                // _ist = (res == 1) ? _imd + 1 : _ist;
                // _ied = (res == 1) ? _ied : _imd;
            }
            #endif
            // printf("ii=%d e_loc_real=%f\n", ii, e_loc_real);
            t_ed = clock();
            clks[2] += static_cast<float64>(t_ed - t_st);
        }

        t_st = clock();
        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
        t_ed = clock();
        clks[3] += static_cast<float64>(t_ed - t_st);
    }
    // printf("tid: %d clks: %.1lf %.1lf %.1f %.1f\n", index, clks[0], clks[1], clks[2], clks[3]);
    // float64 _sum = clks[0] + clks[1] + clks[2] + clks[3];
    // printf("tid: %d clks: %.3lf %.3lf %.3f %.3f\n", index, clks[0]/_sum, clks[1]/_sum, clks[2]/_sum, clks[3]/_sum);
}


template<int TILE_SIZE=4, int BLK_SIZE=128>
__global__ void calculate_local_energy_tile_kernel(
    const int32 n_qubits,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    // const int64 *ks,
    const uint64 *ks,
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;
    const int32 tx = threadIdx.x;

    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    __shared__ dtype state_tile_shm[BLK_SIZE][MAX_NQUBITS*TILE_SIZE];
    __shared__ dtype pauli_mat23_shm[BLK_SIZE][MAX_NQUBITS];
    // loop all samples
    for (int ii = index*TILE_SIZE; ii < batch_size_cur_rank; ii+=stride*TILE_SIZE) {
        psi_dtype e_loc_real[TILE_SIZE] = {0}, e_loc_imag[TILE_SIZE] = {0};
        int REAL_TILE_SIZE = std::min(TILE_SIZE, (int32)batch_size_cur_rank - ii);
        // load state_batch -> state_tile_shm
        for (int j = 0, gj = ii*N; j < N*REAL_TILE_SIZE; j++, gj++) {
            state_tile_shm[tx][j] = state_batch[gj];
        }

        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef[TILE_SIZE] = {0.0};

            // coeff calculation
            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                for (int j = 0; j < N; j++) {
                    pauli_mat23_shm[tx][j] = pauli_mat23[i*N+j];
                }

                // #pragma unroll(4)
                for (int it = 0; it < TILE_SIZE; it++) {
                    int _sum[TILE_SIZE] = {0};
                    for (int ik = 0; ik < N; ik++) {
                        // _sum[it] += state_batch[(ii+it)*N+ik] & pauli_mat23[i_base+ik];
                        // _sum[it] += state_tile_shm[tx][it*N+ik] & pauli_mat23[i*N+ik];
                        _sum[it] += state_tile_shm[tx][it*N+ik] & pauli_mat23_shm[tx][ik];
                    }
                    // if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
                    // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                    const psi_dtype _sgn = (_sum[it] % 2 == 0) ? 1 : -1;
                    coef[it] += _sgn * coeffs[i];
                }
            }

            // coupled state calculation
            for (int it = 0; it < TILE_SIZE; it++) {
                // filter value < eps
                if (FABS(coef[it]) < eps) {
                    continue;
                }

                // map state -> id
                int64 j_base = sid * N;
                uint64 id[TILE_SIZE] = {0};
                for (int ik = 0; ik < N; ik++) {
                    // id[it] += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
                    id[it] += (state_tile_shm[tx][it*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
                }

                // binary find id among the sampled samples
                // idx = binary_find(ks, id), [_ist, _ied) start from 0
                int32 _ist = 0, _ied = batch_size, _imd = 0;
                while (_ist < _ied) {
                    _imd = (_ist + _ied) / 2;
                    if (ks[_imd] == id[it]) {
                        // e_loc += coef * vs[_imid]
                        e_loc_real[it] += coef[it] * vs[_imd * 2];
                        e_loc_imag[it] += coef[it] * vs[_imd * 2 + 1];
                        break;
                    }

                    if (ks[_imd] < id[it]) {
                        _ist = _imd + 1;
                    } else {
                        _ied = _imd;
                    }
                    // int res = ks[_imd] < id;
                    // _ist = (res == 1) ? _imd + 1 : _ist;
                    // _ied = (res == 1) ? _ied : _imd;
                }
            }
        }

        // store the result number as return
        for (int it = 0; it < REAL_TILE_SIZE; it++) {
            // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
            const psi_dtype a = e_loc_real[it], b = e_loc_imag[it];
            const psi_dtype c = vs[(ist+ii+it)*2], d = vs[(ist+ii+it)*2+1];
            const psi_dtype c2_d2 = c*c + d*d;
            res_eloc_batch[(ii+it)*2  ] = (a*c + b*d) / c2_d2;
            res_eloc_batch[(ii+it)*2+1] = -(a*d - b*c) / c2_d2;
        }
    }
}

template<typename T>
__device__ T warp_reduce_sum(T value) {
    #pragma unroll 5
    for (int j = 16; j >= 1; j /= 2)
        value += __shfl_xor_sync(0xffffffff, value, j, 32);
    return value;
}

template<int MAXN=32, int TILE_SIZE=4, int BLK_SIZE=128, int WARP_SIZE=32>
__global__ void calculate_local_energy_warp_tile_nosync_kernel(
// __global__ void calculate_local_energy_warp_tile_kernel(
    const int32 n_qubits,
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
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 tx = threadIdx.x;

    constexpr int32 WARP_NUM = BLK_SIZE / WARP_SIZE;
    const int32 warp_idx = WARP_NUM * blockIdx.x + threadIdx.x / WARP_SIZE;
    const int32 blk_warp_idx = threadIdx.x / WARP_SIZE;
    const int32 stride = gridDim.x * WARP_NUM;
    const int32 lane_id = threadIdx.x % WARP_SIZE;

    // printf("index=%d warp_idx=%d wx=%d WARP_NUM=%d stride=%d\n", index, warp_idx, wx, WARP_NUM, stride);
    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    __shared__ dtype state_tile_shm[WARP_NUM][MAX_NQUBITS*TILE_SIZE];
    __shared__ dtype pauli_mat23_shm[WARP_NUM][MAX_NQUBITS];
    // __shared__ dtype pauli_mat23_shm[MAX_NQUBITS];
    __shared__ dtype pauli_mat12_shm[WARP_NUM][MAX_NQUBITS];
    // __shared__ psi_dtype e_loc_real[TILE_SIZE][WARP_NUM];
    // __shared__ psi_dtype e_loc_imag[TILE_SIZE][WARP_NUM];
    __shared__ psi_dtype e_loc_real[WARP_NUM][TILE_SIZE];
    __shared__ psi_dtype e_loc_imag[WARP_NUM][TILE_SIZE];
    // __shared__ coeff_dtype coef[TILE_SIZE][WARP_NUM];
    __shared__ coeff_dtype coef[WARP_NUM][TILE_SIZE];
    __shared__ coeff_dtype coeffs_shm[WARP_NUM];
    __shared__ uint64 part_id_shm[WARP_NUM][TILE_SIZE];


    // loop all samples
    for (int ii = warp_idx*TILE_SIZE; ii < batch_size_cur_rank; ii+=stride*TILE_SIZE) {
        // psi_dtype e_loc_real[TILE_SIZE] = {0}, e_loc_imag[TILE_SIZE] = {0};
        int REAL_TILE_SIZE = std::min(TILE_SIZE, (int32)batch_size_cur_rank - ii);
        // load state_batch -> state_tile_shm
        // for (int j = 0, gj = ii*N; j < N*REAL_TILE_SIZE; j++, gj++) {
        for (int j = lane_id, gj = ii*N+lane_id; j < N*REAL_TILE_SIZE; j+=WARP_SIZE, gj+=WARP_SIZE) {
            state_tile_shm[blk_warp_idx][j] = state_batch[gj];
        }
        for (int it = lane_id; it < TILE_SIZE; it+=WARP_SIZE) {
            e_loc_real[blk_warp_idx][it] = 0;
            e_loc_imag[blk_warp_idx][it] = 0;
            // e_loc_real[it][blk_warp_idx] = 0;
            // e_loc_imag[it][blk_warp_idx] = 0;
        }
        __syncwarp();
        // if (wx == 0) printf("ii: %d load1_over\n", ii);
        for (int sid = 0; sid < NK; sid++) {
            // coeff_dtype coef[TILE_SIZE] = {0.0};
            for (int it = lane_id; it < TILE_SIZE; it+=WARP_SIZE) {
                // coef[it][blk_warp_idx] = 0;
                coef[blk_warp_idx][it] = 0;
            }
            __syncwarp();
            // TODO: TILE_SIZE <= 32
            // if (wx < TILE_SIZE) coef[wx][blk_warp_idx] = 0;
            // if (wx < TILE_SIZE) coef[blk_warp_idx][wx] = 0;

            // t_st = clock();
            // coeff calculation
            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                if (lane_id == 0) coeffs_shm[blk_warp_idx] = coeffs[i];

                // for (int j = 0; j < N; j++) {
                //     pauli_mat23_shm[warp_idx][j] = pauli_mat23[i*N+j];
                // }
                if (MAXN == 32) {
                    if (lane_id < N) pauli_mat23_shm[blk_warp_idx][lane_id] = pauli_mat23[i*N+lane_id];
                } else if (MAXN == 64) {
                    pauli_mat23_shm[blk_warp_idx][lane_id] = pauli_mat23[i*N+lane_id];
                    int _lane_id = lane_id + WARP_SIZE;
                    if (_lane_id < N) pauli_mat23_shm[blk_warp_idx][_lane_id] = pauli_mat23[i*N+_lane_id];
                }
                __syncwarp();

                //#pragma unroll 4
                for (int it = 0; it < TILE_SIZE; it++) {
                    // int _sum[TILE_SIZE] = {0};
                    // for (int ik = 0; ik < N; ik++) {
                    //     // _sum[it] += state_batch[(ii+it)*N+ik] & pauli_mat23[i_base+ik];
                    //     // _sum[it] += state_tile_shm[tx][it*N+ik] & pauli_mat23[i*N+ik];
                    //     _sum[it] += state_tile_shm[warp_idx][it*N+ik] & pauli_mat23_shm[warp_idx][ik];
                    // }
                    int value = 0;
                    if (MAXN == 32) {
                        if (lane_id < N) value = state_tile_shm[blk_warp_idx][it*N+lane_id] & pauli_mat23_shm[blk_warp_idx][lane_id];
                        value = warp_reduce_sum(value);
                    } else if (MAXN == 64) {
                        value = state_tile_shm[blk_warp_idx][it*N+lane_id] & pauli_mat23_shm[blk_warp_idx][lane_id];
                        value = warp_reduce_sum(value);
                        int _lane_id = lane_id + WARP_SIZE;
                        int value2 = 0;
                        if (_lane_id < N) value2 = state_tile_shm[blk_warp_idx][it*N+_lane_id] & pauli_mat23_shm[blk_warp_idx][_lane_id];
                        value2 = warp_reduce_sum(value2);
                        if (lane_id == 0) {
                            value += value2;
                        }
                    }

                    if (lane_id == 0) {
                        // printf("warp_idx: %d wx: %d value: %d\n", warp_idx, wx, value);
                        const psi_dtype _sgn = (value % 2 == 0) ? 1 : -1;
                        // coef[it] += _sgn * coeffs[i];
                        // coef[it][blk_warp_idx] += _sgn * coeffs[i];
                        // coef[blk_warp_idx][it] += _sgn * coeffs[i];
                        coef[blk_warp_idx][it] += _sgn * coeffs_shm[blk_warp_idx];
                    }
                }
            }
            // t_ed = clock();
            // clks[0] += static_cast<float64>(t_ed - t_st);

            // t_st = clock();
            // if (wx == 0) printf("wx: %d coef_over\n", wx);
            // if (wx == 0) {
            // coupled state calculation
            coeff_dtype cur_coef;
            if (lane_id < TILE_SIZE) part_id_shm[blk_warp_idx][lane_id] = -1;
            for (int it = 0; it < TILE_SIZE; it++) {
                // bcast across warp
                // if (wx == 0) cur_coef = FABS(coef[it]);
                // if (wx == 0) cur_coef = FABS(coef[it][blk_warp_idx]);
                if (lane_id == 0) cur_coef = FABS(coef[blk_warp_idx][it]);
                cur_coef = __shfl_sync(0xffffffff, cur_coef, 0, WARP_SIZE);
                if (cur_coef < eps) {
                    continue;
                }

                // for (int j = 0; j < N; j++) {
                //     pauli_mat12_shm[j] = pauli_mat12[j_base+ik];

                uint64 part_id = 0;

                if (MAXN == 32) {
                    if (lane_id < N) pauli_mat12_shm[blk_warp_idx][lane_id] = pauli_mat12[sid*N+lane_id];
                    if (lane_id < N) part_id = (state_tile_shm[blk_warp_idx][it*N+lane_id] ^ pauli_mat12_shm[blk_warp_idx][lane_id])*tbl_pow2[lane_id];
                    part_id = warp_reduce_sum(part_id);
                } else if (MAXN == 64) {
                    pauli_mat12_shm[blk_warp_idx][lane_id] = pauli_mat12[sid*N+lane_id];
                    part_id = (state_tile_shm[blk_warp_idx][it*N+lane_id] ^ pauli_mat12_shm[blk_warp_idx][lane_id])*tbl_pow2[lane_id];
                    part_id = warp_reduce_sum(part_id);
                    int _lane_id = lane_id + WARP_SIZE;
                    uint64 part_id2 = 0;
                    if (_lane_id < N) pauli_mat12_shm[blk_warp_idx][_lane_id] = pauli_mat12[sid*N+_lane_id];
                    if (_lane_id < N) part_id2 = (state_tile_shm[blk_warp_idx][it*N+_lane_id] ^ pauli_mat12_shm[blk_warp_idx][_lane_id])*tbl_pow2[_lane_id];
                    part_id2 = warp_reduce_sum(part_id2);
                    if (lane_id == 0) {
                        part_id += part_id2;
                    }
                }

                if (lane_id == 0)
                    part_id_shm[blk_warp_idx][it] = part_id;
            }
            __syncwarp();
            
            for (int it = lane_id; it < TILE_SIZE; it += WARP_SIZE) {
                    uint64 id = part_id_shm[blk_warp_idx][it];
                    // if (id == -1) continue;
                    // binary find id among the sampled samples
                    // idx = binary_find(ks, id), [_ist, _ied) start from 0
                    int32 _ist = 0, _ied = batch_size, _imd = 0;
                    while (_ist < _ied) {
                        _imd = (_ist + _ied) / 2;
                        // if (ks[_imd] == id[it]) {
                        if (ks[_imd] == id) {
                            // e_loc += coef * vs[_imid]
                            // e_loc_real[it] += coef[it] * vs[_imd * 2];
                            // e_loc_imag[it] += coef[it] * vs[_imd * 2 + 1];
                            // e_loc_real[it][blk_warp_idx] += coef[it] * vs[_imd * 2];
                            // e_loc_imag[it][blk_warp_idx] += coef[it] * vs[_imd * 2 + 1];
                            // e_loc_real[it][blk_warp_idx] += coef[it][blk_warp_idx] * vs[_imd * 2];
                            // e_loc_imag[it][blk_warp_idx] += coef[it][blk_warp_idx] * vs[_imd * 2 + 1];
                            // e_loc_real[it][blk_warp_idx] += coef[blk_warp_idx][it] * vs[_imd * 2];
                            // e_loc_imag[it][blk_warp_idx] += coef[blk_warp_idx][it] * vs[_imd * 2 + 1];
                            e_loc_real[blk_warp_idx][it] += coef[blk_warp_idx][it] * vs[_imd * 2];
                            e_loc_imag[blk_warp_idx][it] += coef[blk_warp_idx][it] * vs[_imd * 2 + 1];
                            break;
                        }
                        int res = ks[_imd] < id;
                        _ist = (res == 1) ? _imd + 1 : _ist;
                        _ied = (res == 1) ? _ied : _imd;
                    }
            }
            // }
            // t_ed = clock();
            // clks[1] += static_cast<float64>(t_ed - t_st);
        }

        // if (wx == 0) printf("all_over\n");

        
        // store the result number as return
        for (int it = lane_id; it < REAL_TILE_SIZE; it+=WARP_SIZE) {
            // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
            // const psi_dtype a = e_loc_real[it], b = e_loc_imag[it];
            // const psi_dtype a = e_loc_real[it][blk_warp_idx], b = e_loc_imag[it][blk_warp_idx];
            const psi_dtype a = e_loc_real[blk_warp_idx][it], b = e_loc_imag[blk_warp_idx][it];
            // printf("ii: %d it: %d a: %f b: %f\n", ii, it, a, b);
            const psi_dtype c = vs[(ist+ii+it)*2], d = vs[(ist+ii+it)*2+1];
            const psi_dtype c2_d2 = c*c + d*d;
            res_eloc_batch[(ii+it)*2  ] = (a*c + b*d) / c2_d2;
            res_eloc_batch[(ii+it)*2+1] = -(a*d - b*c) / c2_d2;
        }
        
        // printf("tid: %d clks: %.1lf %.1lf %.1f %.1f\n", index, clks[0], clks[1], clks[2], clks[3]);
    }
}

template<int MAXN=32, int TILE_SIZE=4, int BLK_SIZE=128, int WARP_SIZE=32>
// __global__ void calculate_local_energy_warp_tile_sync_kernel(
__global__ void calculate_local_energy_warp_tile_kernel(
    const int32 n_qubits,
    const int64 NK,
    const __restrict__ int64 *idxs,
    const __restrict__ coeff_dtype *coeffs,
    const __restrict__ dtype *pauli_mat12,
    const __restrict__ dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const __restrict__ dtype *state_batch,
    const __restrict__ uint64 *ks,
    const __restrict__ psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 tx = threadIdx.x;

    constexpr int32 WARP_NUM = BLK_SIZE / WARP_SIZE;
    const int32 warp_idx = WARP_NUM * blockIdx.x + threadIdx.x / WARP_SIZE;
    const int32 blk_warp_idx = threadIdx.x / WARP_SIZE;
    const int32 stride = gridDim.x * WARP_NUM;
    const int32 wx = threadIdx.x % WARP_SIZE;

    const int32 ceil_batch_size_cur_rank = WARP_NUM * gridDim.x * TILE_SIZE;
    // printf("index=%d warp_idx=%d wx=%d WARP_NUM=%d stride=%d\n", index, warp_idx, wx, WARP_NUM, stride);
    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    __shared__ dtype state_tile_shm[WARP_NUM][MAX_NQUBITS*TILE_SIZE];
    // __shared__ dtype pauli_mat23_shm[WARP_NUM][MAX_NQUBITS];
    // __shared__ volatile dtype pauli_mat23_shm[WARP_NUM][MAX_NQUBITS];
    // __shared__ volatile dtype pauli_mat23_shm[MAX_NQUBITS];
    __shared__ dtype pauli_mat23_shm[MAX_NQUBITS];
    __shared__ dtype pauli_mat12_shm[WARP_NUM][MAX_NQUBITS];
    // __shared__ psi_dtype e_loc_real[TILE_SIZE][WARP_NUM];
    // __shared__ psi_dtype e_loc_imag[TILE_SIZE][WARP_NUM];
    __shared__ psi_dtype e_loc_real[WARP_NUM][TILE_SIZE];
    __shared__ psi_dtype e_loc_imag[WARP_NUM][TILE_SIZE];
    // __shared__ coeff_dtype coef[TILE_SIZE][WARP_NUM];
    __shared__ coeff_dtype coef[WARP_NUM][TILE_SIZE];
    __shared__ coeff_dtype coeffs_shm[WARP_NUM];

    // float64 clks[4] = {0};
    // clock_t t_st, t_ed;
    // if (index == 0) printf("ceil_batch_size_cur_rank: %d batch_size_cur_rank: %lld\n", ceil_batch_size_cur_rank, batch_size_cur_rank);
    // loop all samples
    for (int ii = warp_idx*TILE_SIZE; ii < batch_size_cur_rank; ii+=stride*TILE_SIZE) {
    // for (int ii = warp_idx*TILE_SIZE; ii < ceil_batch_size_cur_rank; ii+=stride*TILE_SIZE) {
        // psi_dtype e_loc_real[TILE_SIZE] = {0}, e_loc_imag[TILE_SIZE] = {0};
        int REAL_TILE_SIZE = std::min(TILE_SIZE, (int32)batch_size_cur_rank - ii);
        // load state_batch -> state_tile_shm
        for (int j = wx, gj = ii*N+wx; j < N*REAL_TILE_SIZE; j+=WARP_SIZE, gj+=WARP_SIZE) {
            state_tile_shm[blk_warp_idx][j] = state_batch[gj];
        }
        if (wx == 0) {
            for (int it = 0; it < TILE_SIZE; it++) {
                e_loc_real[blk_warp_idx][it] = 0;
                e_loc_imag[blk_warp_idx][it] = 0;
                // e_loc_real[it][blk_warp_idx] = 0;
                // e_loc_imag[it][blk_warp_idx] = 0;
            }
        }

        for (int sid = 0; sid < NK; sid++) {
            // coeff_dtype coef[TILE_SIZE] = {0.0};
            if (wx == 0) {
                for (int it = 0; it < TILE_SIZE; it++) {
                    // coef[it][blk_warp_idx] = 0;
                    coef[blk_warp_idx][it] = 0;
                }
            }
            // TODO: TILE_SIZE <= 32
            // if (wx < TILE_SIZE) coef[wx][blk_warp_idx] = 0;
            // if (wx < TILE_SIZE) coef[blk_warp_idx][wx] = 0;

            // t_st = clock();
            // coeff calculation
            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                if (wx == 0) coeffs_shm[blk_warp_idx] = coeffs[i];
                // for (int j = 0; j < N; j++) {
                //     pauli_mat23_shm[warp_idx][j] = pauli_mat23[i*N+j];
                // }
                // for (int j = wx; j < N; j+=WARP_SIZE) {
                //     pauli_mat23_shm[blk_warp_idx][j] = pauli_mat23[i*N+j];
                // }
                // if (wx < N) pauli_mat23_shm[blk_warp_idx][wx] = pauli_mat23[i*N+wx];
                // if (blk_warp_idx == 0 && wx < N) pauli_mat23_shm[wx] = pauli_mat23[i*N+wx];
                if (tx < N) pauli_mat23_shm[tx] = pauli_mat23[i*N+tx];
                // __syncthreads(); // TODO: bug?
                // if (wx < N) {
                //     // if (pauli_mat23_shm[0][wx] != pauli_mat23[i*N+wx]) {
                //     if (pauli_mat23_shm[wx] != pauli_mat23[i*N+wx]) {
                //         printf("==ERROR1==index: %d wx: %d ii: %d warp_idx: %d blk_warp_idx: %d\n", index, wx, ii, warp_idx, blk_warp_idx);
                //     }
                // }

                #pragma unroll 4
                for (int it = 0; it < TILE_SIZE; it++) {
                    // int _sum[TILE_SIZE] = {0};
                    // for (int ik = 0; ik < N; ik++) {
                    //     // _sum[it] += state_batch[(ii+it)*N+ik] & pauli_mat23[i_base+ik];
                    //     // _sum[it] += state_tile_shm[tx][it*N+ik] & pauli_mat23[i*N+ik];
                    //     _sum[it] += state_tile_shm[warp_idx][it*N+ik] & pauli_mat23_shm[warp_idx][ik];
                    // }
                    int value = 0;
                    __syncthreads(); // TODO: bug?
                    if (MAXN == 32) {
                        if (wx < N) value = state_tile_shm[blk_warp_idx][it*N+wx] & pauli_mat23_shm[wx];
                        value = warp_reduce_sum(value);
                    } else if (MAXN == 64) {
                        value = state_tile_shm[blk_warp_idx][it*N+wx] & pauli_mat23_shm[wx];
                        value = warp_reduce_sum(value);
                        int _wx = wx + WARP_SIZE;
                        int value2 = 0;
                        if (_wx < N) value2 = state_tile_shm[blk_warp_idx][it*N+_wx] & pauli_mat23_shm[_wx];
                        value2 = warp_reduce_sum(value2);
                        if (wx == 0) {
                            value += value2;
                        }
                    }
                    // error check
                    // if (wx < N) {
                    //     // if (pauli_mat23_shm[0][wx] != pauli_mat23[i*N+wx]) {
                    //     if (pauli_mat23_shm[wx] != pauli_mat23[i*N+wx]) {
                    //         printf("==ERROR2==index: %d wx: %d ii: %d warp_idx: %d blk_warp_idx: %d\n", index, wx, ii, warp_idx, blk_warp_idx);
                    //     }
                    // }
                    // value = warp_reduce_sum(value);
                    if (wx == 0) {
                        // printf("warp_idx: %d wx: %d value: %d\n", warp_idx, wx, value);
                        // const psi_dtype _sgn = (value % 2 == 0) ? 1 : -1;
                        // const coeff_dtype _sgn = (value % 2 == 0) ? 1 : -1;
                        // const coeff_dtype _sgn = (value % 2 == 0) * (-2) + 1;
                        // coef[it] += _sgn * coeffs[i];
                        // coef[it][blk_warp_idx] += _sgn * coeffs[i];
                        // coef[blk_warp_idx][it] += _sgn * coeffs[i];
                        // coef[blk_warp_idx][it] += _sgn * coeffs_shm[blk_warp_idx];
                        coef[blk_warp_idx][it] +=  (value & 0x01) ? -coeffs_shm[blk_warp_idx] : coeffs_shm[blk_warp_idx];
                    }
                }
            }
            // t_ed = clock();
            // clks[0] += static_cast<float64>(t_ed - t_st);

            // t_st = clock();
            // coupled state calculation
            coeff_dtype cur_coef;
            for (int it = 0; it < TILE_SIZE; it++) {
                // bcast across warp
                // if (wx == 0) cur_coef = FABS(coef[it][blk_warp_idx]);
                if (wx == 0) cur_coef = FABS(coef[blk_warp_idx][it]);
                cur_coef = __shfl_sync(0xffffffff, cur_coef, 0, WARP_SIZE);
                if (cur_coef < eps) {
                    continue;
                }

                // for (int j = 0; j < N; j++) {
                //     pauli_mat12_shm[j] = pauli_mat12[j_base+ik];
                // }
                // if (wx < N) pauli_mat12_shm[blk_warp_idx][wx] = pauli_mat12[sid*N+wx];
                // map state -> id
                // int64 j_base = sid * N;

                uint64 part_id = 0;
                // if (wx < N) part_id = (state_tile_shm[blk_warp_idx][it*N+wx] ^ pauli_mat12_shm[blk_warp_idx][wx])*tbl_pow2[wx];
                // part_id = warp_reduce_sum(part_id);

                if (MAXN == 32) {
                    if (wx < N) pauli_mat12_shm[blk_warp_idx][wx] = pauli_mat12[sid*N+wx];
                    if (wx < N) part_id = (state_tile_shm[blk_warp_idx][it*N+wx] ^ pauli_mat12_shm[blk_warp_idx][wx])*tbl_pow2[wx];
                    part_id = warp_reduce_sum(part_id);
                } else if (MAXN == 64) {
                    pauli_mat12_shm[blk_warp_idx][wx] = pauli_mat12[sid*N+wx];
                    part_id = (state_tile_shm[blk_warp_idx][it*N+wx] ^ pauli_mat12_shm[blk_warp_idx][wx])*tbl_pow2[wx];
                    part_id = warp_reduce_sum(part_id);
                    int _wx = wx + WARP_SIZE;
                    uint64 part_id2 = 0;
                    if (_wx < N) pauli_mat12_shm[blk_warp_idx][_wx] = pauli_mat12[sid*N+_wx];
                    if (_wx < N) part_id2 = (state_tile_shm[blk_warp_idx][it*N+_wx] ^ pauli_mat12_shm[blk_warp_idx][_wx])*tbl_pow2[_wx];
                    part_id2 = warp_reduce_sum(part_id2);
                    if (wx == 0) {
                        part_id += part_id2;
                    }
                }

                if (wx == 0) {
                    uint64 id = part_id;
                    // binary find id among the sampled samples
                    // idx = binary_find(ks, id), [_ist, _ied) start from 0
                    int32 _ist = 0, _ied = batch_size, _imd = 0;
                    while (_ist < _ied) {
                        _imd = (_ist + _ied) / 2;
                        // if (ks[_imd] == id[it]) {
                        if (ks[_imd] == id) {
                            // e_loc += coef * vs[_imid]
                            // e_loc_real[it] += coef[it] * vs[_imd * 2];
                            // e_loc_imag[it] += coef[it] * vs[_imd * 2 + 1];
                            // e_loc_real[it][blk_warp_idx] += coef[it] * vs[_imd * 2];
                            // e_loc_imag[it][blk_warp_idx] += coef[it] * vs[_imd * 2 + 1];
                            // e_loc_real[it][blk_warp_idx] += coef[it][blk_warp_idx] * vs[_imd * 2];
                            // e_loc_imag[it][blk_warp_idx] += coef[it][blk_warp_idx] * vs[_imd * 2 + 1];
                            // e_loc_real[it][blk_warp_idx] += coef[blk_warp_idx][it] * vs[_imd * 2];
                            // e_loc_imag[it][blk_warp_idx] += coef[blk_warp_idx][it] * vs[_imd * 2 + 1];
                            e_loc_real[blk_warp_idx][it] += coef[blk_warp_idx][it] * vs[_imd * 2];
                            e_loc_imag[blk_warp_idx][it] += coef[blk_warp_idx][it] * vs[_imd * 2 + 1];
                            break;
                        }

                        // if (ks[_imd] < id[it]) {
                        // if (ks[_imd] < id) {
                        //     _ist = _imd + 1;
                        // } else {
                        //     _ied = _imd;
                        // }
                        int res = ks[_imd] < id;
                        _ist = (res == 1) ? _imd + 1 : _ist;
                        _ied = (res == 1) ? _ied : _imd;
                    }
                }
            }
            // }
            // t_ed = clock();
            // clks[1] += static_cast<float64>(t_ed - t_st);
        }

        // if (wx == 0) printf("all_over\n");

        if (wx == 0) {
            // store the result number as return
            for (int it = 0; it < REAL_TILE_SIZE; it++) {
                // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
                // const psi_dtype a = e_loc_real[it], b = e_loc_imag[it];
                // const psi_dtype a = e_loc_real[it][blk_warp_idx], b = e_loc_imag[it][blk_warp_idx];
                const psi_dtype a = e_loc_real[blk_warp_idx][it], b = e_loc_imag[blk_warp_idx][it];
                // printf("ii: %d it: %d a: %f b: %f\n", ii, it, a, b);
                const psi_dtype c = vs[(ist+ii+it)*2], d = vs[(ist+ii+it)*2+1];
                const psi_dtype c2_d2 = c*c + d*d;
                res_eloc_batch[(ii+it)*2  ] = (a*c + b*d) / c2_d2;
                res_eloc_batch[(ii+it)*2+1] = -(a*d - b*c) / c2_d2;
            }
        }
        // printf("tid: %d clks: %.1lf %.1lf %.1f %.1f\n", index, clks[0], clks[1], clks[2], clks[3]);
    }
}


template<int MAXN=64>
__global__ void calculate_local_energy_kernel_bigInt_V1_bitarr_shm(
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
    psi_dtype *res_eloc_batch)
{
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;
    int lane_id = threadIdx.x % 32;
    __shared__ dtype sh_state [128*3];//TODO dynamic shared memory size;

    uint64 big_id[ID_WIDTH];

    // loop all samples
    for (int ii = index; ii < (batch_size_cur_rank+31)/32*32; ii+=stride) {
        //__syncwarp();
        int active_thread_num = batch_size_cur_rank - ((ii >> 5) << 5);
        active_thread_num = (active_thread_num > 32) ? 32 : active_thread_num;
        int read_base_off = ((ii >> 5) << 5)*N;
        int write_base_off = (threadIdx.x-lane_id)*N;
        for (int idx = lane_id; idx < active_thread_num*N; idx += 32) {
            sh_state[write_base_off + idx] = state_batch[read_base_off + idx];
        }
        //__threadfence();
        read_base_off -= (threadIdx.x-lane_id)*N;
        if (ii >= batch_size_cur_rank)
            continue;
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;
            // map state -> id
            // int64 j_base = sid * N;
            int res = 0xffff;
            // int64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            //     id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            // }
            // _state2id_huge_fuse(&state_batch[ii*N], &pauli_mat12[j_base], N, id_width, id_stride, tbl_pow2, big_id);
            big_id[0] = sh_state[ii*N - read_base_off] ^ pauli_mat12[sid*N];
            if (MAXN >= 64) {
                big_id[0] = ((uint64)(sh_state[ii*N+1 - read_base_off] ^ pauli_mat12[sid*N+1]) << 32) | big_id[0];
            }
            if (MAXN >= 96) {
                big_id[1] = sh_state[ii*N+2 - read_base_off] ^ pauli_mat12[sid*N+2];
            }
            if (MAXN >= 128) {
                big_id[1] = ((uint64)(sh_state[ii*N+3 - read_base_off] ^ pauli_mat12[sid*N+3]) << 32) | big_id[1];
            }
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size;
            binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            // printf("index: %d big_id[0]: %llu res: %d\n", index, big_id[0], res);

            // don't find this coupled state in current samples
            if (res != 0) {
                continue;
            }

            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                // int _sum = 0;
                // for (int ik = 0; ik < N; ik++) {
                //     // _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                //     _sum += state_batch[ii*N+ik] & pauli_mat23[i*N+ik];
                // }
                int _sum = __popc(sh_state[ii*N - read_base_off] & pauli_mat23[i*N]);
                if (MAXN >= 64) {
                    _sum += __popc(sh_state[ii*N+1 - read_base_off] & pauli_mat23[i*N+1]);
                }
                if (MAXN >= 96) {
                    _sum += __popc(sh_state[ii*N+2 - read_base_off] & pauli_mat23[i*N+2]);
                }
                if (MAXN >= 128) {
                    _sum += __popc(sh_state[ii*N+3 - read_base_off] & pauli_mat23[i*N+3]);
                }
                // if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                // i_base += N;
            }
            // if (FABS(coef) < eps) continue;
            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;

            // printf("ii: %d coef: %f\n", ii, coef);
            // printf("ii=%d e_loc_real=%f\n", ii, e_loc_real);
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
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
    psi_dtype *res_eloc_batch)
{
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    uint64 big_id[ID_WIDTH];

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;
            // map state -> id
            // int64 j_base = sid * N;
            int res = 0xffff;
            // int64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            //     id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            // }
            // _state2id_huge_fuse(&state_batch[ii*N], &pauli_mat12[j_base], N, id_width, id_stride, tbl_pow2, big_id);
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
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size;
            binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            // printf("index: %d big_id[0]: %llu res: %d\n", index, big_id[0], res);

            // don't find this coupled state in current samples
            if (res != 0) {
                continue;
            }

            coeff_dtype coef = 0.0;

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
            // if (FABS(coef) < eps) continue;
            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;

            // printf("ii: %d coef: %f\n", ii, coef);
            // printf("ii=%d e_loc_real=%f\n", ii, e_loc_real);
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

__global__ void calculate_local_energy_kernel_V1(
    const int32 n_qubits,
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
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    // const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    // const int32 stride = gridDim.x * blockDim.x;
    const int32 thread_index = threadIdx.x;
    const int32 thread_stride = blockDim.x;
    const int32 block_index = blockIdx.x;
    const int32 block_stride = gridDim.x;
    // const int32 idx = threadIdx.x + threadIdx.y * blockDim.x;
    // if (!blockIdx.x && !threadIdx.x) printf("gridDim.x=%d blockDim.x=%d\n", gridDim.x, blockDim.x);
    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }
    __shared__ dtype shm_state[MAX_NQUBITS];
    __shared__ coeff_dtype shm_coefs[32];
    // loop all samples
    // for (int ii = 0; ii < batch_size_cur_rank; ii++) {
    // for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
    for (int ii = block_index; ii < batch_size_cur_rank; ii+=block_stride) {
        __syncthreads(); // avoid shm_state write first
        if (!threadIdx.y) {
            for (int ik = thread_index; ik < N; ik+=thread_stride) {
                shm_state[ik] = state_batch[ii*N+ik];
            }
        }
        __syncthreads();
        // if (!block_index&&!idx) printf("blk=0, idx=0 shm_state[0]=%d\n", shm_state[0]);
        // if (!idx) printf("ii:%d  %d %d %d %d\n", ii, shm_state[0], shm_state[1], shm_state[2], shm_state[3]);
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        // int64 i_base = 0;
        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            // for (int i = st; i < ed; i++) {
            for (int i = st+threadIdx.y; i < ed; i+=blockDim.y) {
                int _sum = 0;
                for (int ik = thread_index; ik < N; ik+=thread_stride) {
                    // _sum += shm_state[ik] & pauli_mat23[i_base+ik];
                    _sum += shm_state[ik] & pauli_mat23[i*N+ik];
                }
                __syncwarp();

                // warp-shuffle reduction
                for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
                    _sum += __shfl_down_sync(0xFFFFFFFF, _sum, offset);
                }
                
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                if (!threadIdx.x) {
                    // if (ii==0) printf("i: %d thd.y: %d _sum: %d\n", i, threadIdx.y, _sum);
                    const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                    coef += _sgn * coeffs[i];
                }
                // __syncwarp();
                // i_base += N;
                // i_base += N*blockDim.y;
                // i_base = N*i;
            }
            // if (!block_index&&!idx) printf("blk=0, idx=0 coef=%f\n", coef);
            // if (ii==0&&!threadIdx.x) printf("Aefore sum: ii=0, thd.y=%d coef=%f\n", threadIdx.y, coef);
            if (!threadIdx.x) shm_coefs[threadIdx.y] = coef;
            __syncthreads();
            // if (!block_index&&!idx) printf("blk=0, idx=0 after sync1: coef=%f\n", coef);
            if (!threadIdx.y && !threadIdx.x) {
                coef = 0.0;
                for (int ic = 0; ic < blockDim.y; ic++) {
                    coef += shm_coefs[ic];
                }
            }
            if (!threadIdx.y) {
                coef = __shfl_sync(0xFFFFFFFF, coef, 0);
            }
            __syncthreads();
            // if (!idx) printf("After sum: ii=%d idx=0 coef=%f\n", ii, coef);
            // if (idx==1) printf("idx=1 coef=%f\n", coef);
            // if (idx==32) printf("idx=32 coef=%f\n", coef);
            // if (threadIdx.x==1) printf("sync coef=%f\n", coef);
            if (threadIdx.y) continue;
            // filter value < eps
            if (FABS(coef) < eps) {
                continue;
            }

            // if (!idx) printf("tid 0 coef is valid\n");

            // map state -> id
            int64 j_base = sid * N;
            uint64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            for (int ik = thread_index; ik < N; ik+=thread_stride) {
                // id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
                id += (shm_state[ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            }
            __syncwarp();
            // warp-shuffle reduction
            for (int offset = warpSize>>1; offset > 0; offset >>= 1) {
                id += __shfl_down_sync(0xFFFFFFFF, id, offset);
            }

            if (threadIdx.x) continue;
            // if (!threadIdx.x) printf("tid 0 enter binary find, id=%ld\n", id);
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size, _imd = 0;
            while (_ist < _ied) {
                _imd = (_ist + _ied) / 2;
                if (ks[_imd] == id) {
                    // e_loc += coef * vs[_imid]
                    e_loc_real += coef * vs[_imd * 2];
                    e_loc_imag += coef * vs[_imd * 2 + 1];
                    break;
                }

                if (ks[_imd] < id) {
                    _ist = _imd + 1;
                } else {
                    _ied = _imd;
                }
            }
            // if (!idx) printf("ii=%d tid 0 finish binary find, e_loc_real=%f\n", ii, e_loc_real);
        }
        if (threadIdx.x || threadIdx.y) continue;
        // if (!idx) printf("samples ii=%d\n", ii);
        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

void calculate_local_energy_sampling_parallel(
    const int64 all_batch_size,
    const int64 batch_size,
    const int64 *_states,
    const int64 ist,
    const int64 ied,
    const int64 ks_disp_idx,
    // const int64 *ks,
    const uint64_t *ks,
    const psi_dtype *vs,
    const int64 rank,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    const int64 target_value = -1;
    const size_t size_states = sizeof(dtype) * batch_size * N;
    dtype *states = NULL, *d_states = NULL;
    states = (dtype *)malloc(size_states);
    memset(states, 0, size_states); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            if (_states[i*N+j] != target_value) {
                states[i*N+j] = 1;
            }
        }
    }

    const size_t size_ks = sizeof(uint64_t) * all_batch_size;
    const size_t size_vs = sizeof(psi_dtype) * all_batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64_t *d_ks = NULL;
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

    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
    calculate_local_energy_kernel<<<nblocks, nthreads>>>(

    // const int nthreads = 32;
    // // const int nblocks = MIN(108, batch_size_cur_rank);
    // const int nblocks = batch_size_cur_rank;

    // const dim3 nthreads(32, 32);
    // const int nblocks = batch_size_cur_rank;
    // calculate_local_energy_kernel_V1<<<nblocks, nthreads>>>(
        n_qubits,
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
        d_vs,
        eps,
        d_res_eloc_batch);
    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
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
    nvtxRangePushA("lck");
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
            d_res_eloc_batch);
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
            d_res_eloc_batch);
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
            d_res_eloc_batch);
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
            d_res_eloc_batch);
    } else {
        printf("Error: only support n_qubits <= 128\n");
    }
    nvtxRangePop();
    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[2].stop("bigIntBitarray: local_energy_kernel");

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
}
