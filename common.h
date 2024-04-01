#pragma once

#define USE_CUDA 1
#define USE_BVH 1
#define PRINT_LOG 1
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#if USE_CUDA
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define CUDA_CALLABLE __device__ __host__
#define CUDA_GLOBAL __global__
#define CUDA_MALLOC(ptr, size, type) cudaMallocManaged(&ptr, size)
#define CUDA_FREE(ptr) cudaFree(ptr)
#define CUDA_MEMSET(ptr, value, size) cudaMemset(ptr, value, size)

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
#define CUDA_MALLOC(ptr, size, type) ptr = new type[size]
#define CUDA_FREE(ptr) delete[] ptr
#define CUDA_MEMSET(ptr, value, size) memset(ptr, value, size)
#define CUDA_LOG(...)

class BaseClass
{
};
#endif

