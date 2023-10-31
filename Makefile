LIBNAME=libcs.so
CPU_SRC=coupled_states.cpp
GPU_SRC=coupled_states.cu
GPU_SRC_DEV=coupled_states_dev.cu
TEST_GPU_SRC=test_coupled_states.cu

VERSION=12.1
CUDA_ROOT=/usr/local/cuda

NVCC=${CUDA_ROOT}/bin/nvcc
LIB_DIR=${CUDA_ROOT}/lib64/
OPTIONS=--std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -arch=compute_86 -code=sm_86
#OPTIONS=--std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -arch=compute_70 -code=sm_80 --ptxas-options=-v 
#INCLUDES=-I/root/wuyangjun/cuCollections/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/compilers/include/ -I./
INCLUDES=-I./
all: test-bitarr

cpu: ${CPU_SRC}
	g++ -fPIC -shared -O3 $< -o ${LIBNAME}

cpu-openmp: ${CPU_SRC}
	g++ -fPIC -shared -O3 $< -o ${LIBNAME} -fopenmp

gpu: ${GPU_SRC_DEV}
	${NVCC} ${OPTIONS} -D BIT_ARRAY_OPT ${INCLUDES} -Xcompiler "-fPIC -shared" $< -O3 -lineinfo -o ${LIBNAME} -L${LIB_DIR}

gpu-tile: ${GPU_SRC_DEV}
	@#${NVCC} ${OPTIONS} -D TILE_OPT ${INCLUDES} -Xcompiler "-fPIC -shared" $< -O3 -lineinfo -o ${LIBNAME} -L${LIB_DIR}
	${NVCC} ${OPTIONS} -D WARP_TILE_OPT ${INCLUDES} -Xcompiler "-fPIC -shared" $< -O3 -lineinfo -o ${LIBNAME} -L${LIB_DIR}

gpu-base: ${GPU_SRC}
	${NVCC} ${OPTIONS} ${INCLUDES} -Xcompiler "-fPIC -shared" $< -O3 -lineinfo -g -o ${LIBNAME} -L${LIB_DIR}

gpu-base-dump: ${GPU_SRC}
	${NVCC} ${OPTIONS} -D NPY_SAVE_DATA ${INCLUDES} -Xcompiler "-fPIC -shared" $< -O3 -lineinfo -g -o ${LIBNAME} -L${LIB_DIR}

test: ${TEST_GPU_SRC}
	${NVCC} ${OPTIONS} test_coupled_states.cu -o gpu_test

test-bitarr: ${TEST_GPU_SRC}
	${NVCC} ${OPTIONS} -D BIT_ARRAY_OPT test_coupled_states.cu -o gpu_test_bitarray -lineinfo -L/usr/local/cuda/lib64 -lnvToolsExt # -g -G

clean:
	rm *.o *.so -f
