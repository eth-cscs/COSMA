#pragma once
#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mutex>

using value_type = double;
using size_type  = size_t;

// helper for initializing cublas
// use only for demos: not threadsafe
static cublasHandle_t get_cublas_handle() {
    static bool is_initialized = false;
    static cublasHandle_t cublas_handle;

    if(!is_initialized) {
        cublasCreate(&cublas_handle);
    }
    return cublas_handle;
}

///////////////////////////////////////////////////////////////////////////////
// CUDA error checking
///////////////////////////////////////////////////////////////////////////////
static void cuda_check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
        << cudaGetErrorString(status) << std::endl;
        throw(std::runtime_error("CUDA ERROR"));
    }
}

static void cublas_check_status(cublasStatus_t status) {
    if(status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "error: CUDABLAS API call\n";
        throw(std::runtime_error("CUDA ERROR"));
    }
}

static void cuda_check_last_kernel(std::string const& errstr) {
    auto status = cudaGetLastError();
    if(status != cudaSuccess) {
        std::cout << "error: CUDA kernel launch : " << errstr << " : "
        << cudaGetErrorString(status) << std::endl;
        throw(std::runtime_error("CUDA ERROR"));
    }
}

///////////////////////////////////////////////////////////////////////////////
// allocating memory
///////////////////////////////////////////////////////////////////////////////

// allocate space on GPU for n instances of type T
template <typename T>
T* malloc_device(size_t n) {
    std::cout << "Allocating " << n << " on device " << std::endl;
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    cuda_check_status(status);
    return (T*)p;
}

// allocate managed memory
template <typename T>
T* malloc_managed(size_t n, T value=T()) {
    std::cout << "Allocating " << n << " on managed " << std::endl;
    T* p;
    auto status = cudaMallocManaged(&p, n*sizeof(T));
    cuda_check_status(status);
    std::fill(p, p+n, value);
    return p;
}

template <typename T>
T* malloc_pinned(size_t N, T value=T()) {
    std::cout << "Allocating " << N << " pinned " << std::endl;
    auto status = cudaGetLastError();
    cuda_check_status(status);
    std::cout << "After get last error " << std::endl;
    T* ptr = nullptr;
    status = cudaHostAlloc((void**)&ptr, N*sizeof(T), 0);
    cuda_check_status(status);
    std::fill(ptr, ptr+N, value);
    return ptr;
}

///////////////////////////////////////////////////////////////////////////////
// copying memory
///////////////////////////////////////////////////////////////////////////////

// copy n*T from host to device
template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyHostToDevice);
}

// copy n*T from device to host
template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyDeviceToHost);
}

// copy n*T from host to device
// If a cuda stream is passed as the final argument the copy will be performed
// asynchronously in the specified stream, otherwise it will be serialized in
// the default (NULL) stream
template <typename T>
void copy_to_device_async(const T* from, T* to, size_t n, cudaStream_t stream=NULL) {
    cudaDeviceSynchronize();
    auto status = cudaGetLastError();
    if(status != cudaSuccess) {
        std::cout << "error: CUDA kernel launch:"
        << cudaGetErrorString(status) << std::endl;
        exit(1);
    }

    status =
    cudaMemcpyAsync(to, from, n*sizeof(T), cudaMemcpyHostToDevice, stream);
    cuda_check_status(status);
}

// copy n*T from device to host
// If a cuda stream is passed as the final argument the copy will be performed
// asynchronously in the specified stream, otherwise it will be serialized in
// the default (NULL) stream
template <typename T>
void copy_to_host_async(const T* from, T* to, size_t n, cudaStream_t stream=NULL) {
    auto status =
    cudaMemcpyAsync(to, from, n*sizeof(T), cudaMemcpyDeviceToHost, stream);
    cuda_check_status(status);
}

