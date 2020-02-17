#include <algorithm>
#include <cosma/local_multiply.hpp>

#include <cublasXt.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <Tiled-MM/util.hpp>
#include <Tiled-MM/mm_handle.hpp>
#include <random>

#include <chrono>
#include <vector>
#include <iostream>

template <typename T>
void fill_matrix(T* ptr, size_t size) {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_real_distribution<T> dist(10.0); // distribution

    for (unsigned i = 0; i < size; ++i) {
        ptr[i] = T{dist(rng)};
    }
}

std::vector<long> tiled_mm_dgemm(int n_iter, int m, int n, int k) {
    auto gpu_ctx = gpu::make_context<double>(2, 4000, 4000, 4000);

    std::vector<double> aa(m * k);
    std::vector<double> bb(k * n);
    std::vector<double> cc(m * n);

    double *a = aa.data();
    double *b = bb.data();
    double *c = cc.data();

    double alpha = 1.0;
    double beta = 0.0;

    std::vector<long> times(n_iter);
    for (int i = 0; i < n_iter; ++i) {
        fill_matrix(a, aa.size());
        fill_matrix(b, bb.size());
        if (beta > 0) {
            fill_matrix(c, cc.size());
        }

        // perform dgemm
        auto start = std::chrono::steady_clock::now();
        cosma::local_multiply(gpu_ctx.get(), a, b, c, m, n, k, alpha, beta);
        auto end = std::chrono::steady_clock::now();

        times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    std::sort(times.begin(), times.end());
    return times;
}

std::vector<long> cublasXt_dgemm(int n_iter, int m, int n, int k) {
    auto status=
    cudaSetDevice(0);
    gpu::check_runtime_status(status);

    cublasXtHandle_t handle;
    auto cublas_status = cublasXtCreate(&handle);
    gpu::check_blas_status(cublas_status);
    int devices[1] = {0};
    cublasXtDeviceSelect(handle, 1, devices);
    // cublasXtSetCpuRoutine(handle, CUBLASXT_GEMM, CUBLASXT_DOUBLE, (void*)(&dgemm_));
    // cublasXtSetCpuRatio(handle, CUBLASXT_GEMM, CUBLASXT_DOUBLE, 0.2);
    // cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED);
    // cublasXtSetBlockDim(handle, 4000);

    std::vector<double> aa(m * k);
    std::vector<double> bb(k * n);
    std::vector<double> cc(m * n);

    double *a = aa.data();
    double *b = bb.data();
    double *c = cc.data();

    double alpha = 1.0;
    double beta = 0.0;

    std::vector<long> times(n_iter);

    for (int i = 0; i < n_iter; ++i) {
        fill_matrix(a, aa.size());
        fill_matrix(b, bb.size());
        if (beta > 0) {
            fill_matrix(c, cc.size());
        }

        // perform dgemm
        auto start = std::chrono::steady_clock::now();
        auto status = cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, &alpha, a, m, b, k, &beta, c, m);
        gpu::check_blas_status(status);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();

        times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    std::sort(times.begin(), times.end());

    // finalization 
    if (handle)
        cublasXtDestroy(handle);

    return times;
}

/*
// cublasLt assumes device pointers
std::vector<long> cublasLt_dgemm(int n_iter, int m, int n, int k) {
    auto runtime_status=
    cudaSetDevice(0);
    gpu::check_runtime_status(runtime_status);

    cublasLtHandle_t handle;
    auto status = cublasLtCreate(&handle);
    gpu::check_blas_status(status);
    // int devices[1] = {0};
    // cublasLtDeviceSelect(handle, 1, devices);

    // std::vector<double> aa(m * k);
    // std::vector<double> bb(k * n);
    // std::vector<double> cc(m * n);

    double *a = gpu::malloc_device<double>(m * k);
    double *b = gpu::malloc_device<double>(k * n);
    double *c = gpu::malloc_device<double>(m * n);

    double alpha = 1.0;
    double beta = 0.0;

    std::size_t workspaceSize = 4000;
    std::size_t workspaceSizeBytes = workspaceSize * sizeof(double);
    auto workspace = gpu::malloc_device<double>(workspaceSize);

    auto transa = CUBLAS_OP_N;
    auto transb = CUBLAS_OP_N;

    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr;
    cublasLtMatrixLayout_t Bdesc = nullptr;
    cublasLtMatrixLayout_t Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    status = cublasLtMatmulDescCreate(&operationDesc, CUDA_R_64F);
    gpu::check_blas_status(status);

    status = cublasLtMatmulDescSetAttribute(operationDesc, 
            CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    gpu::check_blas_status(status);
    status = cublasLtMatmulDescSetAttribute(operationDesc, 
            CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    gpu::check_blas_status(status);

    status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, m, k, m);
    gpu::check_blas_status(status);
    status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, k, n, k);
    gpu::check_blas_status(status);
    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, m);
    gpu::check_blas_status(status);

    std::cout << "Created matrix layouts." << std::endl;

    status = cublasLtMatmulPreferenceCreate(&preference);
    gpu::check_blas_status(status);

    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
        &workspaceSizeBytes, sizeof(workspaceSizeBytes));
    gpu::check_blas_status(status);

    std::cout << "Set up preferences." << std::endl;

    status = cublasLtMatmulAlgoGetHeuristic(
        handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, 
        preference, 1, &heuristicResult, &returnedResults);
    gpu::check_blas_status(status);

    if (returnedResults == 0) {
        std::cout << "No algorithm was returned." << std::endl;
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        gpu::check_blas_status(status);
    }

    std::cout << "Chose the algorithm." << std::endl;

    std::vector<long> times(n_iter);

    for (int i = 0; i < n_iter; ++i) {
        // fill_matrix(a, m * k);
        // fill_matrix(b, k * n);
        // if (beta > 0) {
        //     fill_matrix(c, m * n);
        // }

        // perform dgemm
        auto start = std::chrono::steady_clock::now();
        status = cublasLtMatmul(handle,
                               operationDesc,
                               &alpha,
                               a,
                               Adesc,
                               b,
                               Bdesc,
                               &beta,
                               c,
                               Cdesc,
                               c,
                               Cdesc,
                               &heuristicResult.algo,
                               workspace,
                               workspaceSizeBytes,
                               0);
        gpu::check_blas_status(status);
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();

        times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    std::sort(times.begin(), times.end());

    // finalization 
    if (handle)
        cublasLtDestroy(handle);
    // Descriptors are no longer needed as all GPU work was already
   // enqueued.
   if (preference) 
       status = cublasLtMatmulPreferenceDestroy(preference);
   if (Cdesc) 
       status = cublasLtMatrixLayoutDestroy(Cdesc);
   if (Bdesc) 
       status = cublasLtMatrixLayoutDestroy(Bdesc);
   if (Adesc) 
       status = cublasLtMatrixLayoutDestroy(Adesc);
   if (operationDesc) 
       status = cublasLtMatmulDescDestroy(operationDesc);
    gpu::check_blas_status(status);

    return times;
}
*/

int main(int argc, char* argv[]) {
    // std::vector<int> dims = {500, 1000, 2000, 4000, 8000, 16000, 32000};
    std::vector<int> dims = {4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000};
    int n_iter = 2;

    std::vector<long> times(n_iter);

    for (const int& dim : dims) {
        std::cout << "Dimension = " << dim << std::endl;
        /*
        // cublasLt
        times = cublasLt_dgemm(n_iter, dim, dim, dim);
        std::cout << "cublasLt: ";
        for (const auto& time : times) {
            std::cout << time << ", ";
        }
        std::cout << std::endl;
        */

        // cublasXt
        times = cublasXt_dgemm(n_iter, dim, dim, dim);
        std::cout << "cublasXt: ";
        for (const auto& time : times) {
            std::cout << time << ", ";
        }
        if (times.size()) {
            std::cout << "highest throughtput [Glop/s]: " << 2.0*dim*dim*dim/(1e6*times[0]);
        }
        std::cout << std::endl;

        // tiled-mm
        times = tiled_mm_dgemm(n_iter, dim, dim, dim);
        std::cout << "Tiled-MM: ";
        for (const auto& time : times) {
            std::cout << time << ", ";
        }
        if (times.size()) {
            std::cout << "highest throughtput [Glop/s]: " << 2.0*dim*dim*dim/(1e6*times[0]);
        }
        std::cout << std::endl;
    }

}
