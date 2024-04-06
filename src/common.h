#pragma once

#define USE_CUDA 1
#define USE_BVH 1
#define PRINT_LOG 0
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

constexpr int MAX_DEPTH = 20;

#if USE_CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#define CUDA_CALLABLE __device__ __host__
#define CUDA_GLOBAL __global__
#define CUDA_DEVICE __device__
#define CUDA_MALLOC(ptr, size, type) cudaMallocManaged(&ptr, size)
#define CUDA_FREE(ptr) cudaFree(ptr)
#define CUDA_MEMSET(ptr, value, size) cudaMemset(ptr, value, size)
#define CURAND_STATE_T_PTR curandState_t*
#define CUDA_CHECK(err) {cudaError_t e = err; if (e != cudaSuccess) {printf("Cuda error: %s\n", cudaGetErrorString(e)); exit(1);}}
#define STACK_SIZE_BVH (1024 * 8)
#define STACK_SIZE_CASTRAY (512)
constexpr size_t STACK_SIZE = (MAX_DEPTH * STACK_SIZE_CASTRAY + STACK_SIZE_BVH);

#if PRINT_LOG
#define CUDA_LOG(...) {printf(__VA_ARGS__);}
#else
#define CUDA_LOG(...)
#endif

class BaseClass
{
public:
    void* operator new(size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaFree(ptr);
    }

    void* operator new[](size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    void operator delete[](void* ptr) noexcept {
        cudaFree(ptr);
    }
};


#else
#define CUDA_CALLABLE
#define CUDA_GLOBAL
#define CUDA_DEVICE
#define CUDA_MALLOC(ptr, size, type) ptr = new type[size]
#define CUDA_FREE(ptr) delete[] ptr
#define CUDA_MEMSET(ptr, value, size) memset(ptr, value, size)
#define CUDA_LOG(...)
#define CURAND_STATE_T_PTR char
#include <random>

class BaseClass
{
};
#endif

