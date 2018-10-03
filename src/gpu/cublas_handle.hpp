#pragma once

#include "util.hpp"

// wrapper around cublasHandle
class cublas_handle {
public:
    cublas_handle() {
        cudaSetDevice(0);
        cublasCreate(&handle_);
        cuda_check_last_kernel("cublasCreate");
        valid_ = true;
    }

    ~cublas_handle() {
        if (valid_) {
            cublasDestroy(handle_);
            cuda_check_last_kernel("cublasDestroy");
        }
    }

    // move constructor
    cublas_handle(cublas_handle&& other) {
        handle_ = other.handle_;
        valid_ = other.valid_;
        other.valid_ = false;
    }

    // move-assignment operator
    cublas_handle& operator=(cublas_handle&& other) {
        if (this != &other) {
            if (valid_) {
                cublasDestroy(handle_);
                cuda_check_last_kernel("cublasDestroy");
            }
            handle_ = other.handle_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    // copy-constructor disabled
    cublas_handle(cublas_handle&) = delete;
    // copy-operator disabled
    cublas_handle& operator=(cublas_handle&) = delete;

    // return the unerlying cublas handle
    cublasHandle_t& handle() {
        return handle_;
    }

private:
    bool valid_ = false;
    cublasHandle_t handle_;
};

static cublasHandle_t get_cublas_handle(int index) {
    static std::vector<cublas_handle> handles(N_STREAMS);
    return handles[index].handle();
}
