#include <cosma/multiply.hpp>

#include <cublasXt.h>
#include <cuda_runtime_api.h>
#include <gpu/util.hpp>

#include <chrono>
#include <vector>
#include <iostream>

long cublas_dgemm(cublasXtHandle_t& handle, int m, int n, int k) {

    std::vector<double> aa(m * k);
    std::vector<double> bb(k * n);
    std::vector<double> cc(m * n);

    double *a = aa.data();
    double *b = bb.data();
    double *c = cc.data();

    double alpha = 1.0;
    double beta = 0.0;

    // perform dgemm
    auto start = std::chrono::steady_clock::now();
    auto status = cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha, a, m, b, k, &beta, c, m);
    cosma::gpu::cublas_check_status(status);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
    auto status=
    cudaSetDevice(0);
    cosma::gpu::cuda_check_status(status);

    cublasXtHandle_t handle;
    auto cublas_status = cublasXtCreate(&handle);
    cosma::gpu::cublas_check_status(cublas_status);
    int devices[1] = {0};
    cublasXtDeviceSelect(handle, 1, devices);
    // cublasXtSetCpuRoutine(handle, CUBLASXT_GEMM, CUBLASXT_DOUBLE, (void*)(&dgemm_));
    // cublasXtSetCpuRatio(handle, CUBLASXT_GEMM, CUBLASXT_DOUBLE, 0.2);
    cublasXtSetPinningMemMode(handle, CUBLASXT_PINNING_ENABLED);
    cublasXtSetBlockDim(handle, 4000);
    // std::vector<int> dims = {500, 1000, 2000, 4000, 8000, 16000, 32000};
    std::vector<int> dims = {32000};
    int n_iter = 1;

    for (const int& dim : dims) {
        std::cout << "Dimension = " << dim << std::endl;
        double t_avg_cublas = 0;

        for (int i = 0; i < n_iter+1; ++i) {
            long t_cublas = cublas_dgemm(handle, dim, dim, dim);

            if (i == 0) continue;

            t_avg_cublas += t_cublas;
        }
        std::cout << "cublas average time [ms]: " << 1.0*t_avg_cublas/n_iter << std::endl;
    }

    // finalization 
    if (handle)
        cublasXtDestroy(handle);

}
