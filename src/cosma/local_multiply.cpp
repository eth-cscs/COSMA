#include <cosma/local_multiply.hpp>
#include <cosma/profiler.hpp>
#include <cosma/timer.hpp>

#ifdef COSMA_HAVE_GPU
#include <Tiled-MM/tiled_mm.hpp>
#include <Tiled-MM/util.hpp>
#endif

#if defined(COSMA_WITH_BLAS) || defined(COSMA_WITH_MKL_BLAS)
#include <cosma/blas.hpp>
#endif

#include <chrono>
#include <complex>
#include <vector>

#include <mpi.h>

namespace cosma {

using clock_t = std::chrono::high_resolution_clock;
using ms_t = std::chrono::milliseconds;

template <typename Scalar>
void print_matrix(int m, int n, Scalar *A, char label) {
    std::cout << "Matrix " << label << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[j * m + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename Scalar>
clock_t::time_point debug_gemm_start(Scalar *matrixA,
                                     Scalar *matrixB,
                                     Scalar *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     Scalar alpha,
                                     Scalar beta) {
    auto start = clock_t::now();
    if (std::abs(beta) > 0) {
        std::cout << "C (before) = " << std::endl;
        print_matrix(m, n, matrixC, 'C');
        auto C_partial = std::unique_ptr<Scalar[]>(new Scalar[m * n]);
        gemm(m, n, k, alpha, matrixA, m, matrixB, k, 0.0, C_partial.get(), m);
        std::cout << "C (partial) = " << std::endl;
        print_matrix(m, n, C_partial.get(), 'C');
    }
    return start;
}

template <typename Scalar>
clock_t::time_point debug_gemm_end(Scalar *matrixA,
                                   Scalar *matrixB,
                                   Scalar *matrixC,
                                   int m,
                                   int n,
                                   int k,
                                   Scalar alpha,
                                   Scalar beta) {
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    print_matrix(m, k, matrixA, 'A');
    print_matrix(k, n, matrixB, 'B');
    print_matrix(m, n, matrixC, 'C');

    return std::chrono::high_resolution_clock::now();
}

#ifdef COSMA_HAVE_GPU
template <typename Scalar>
void local_multiply(gpu::mm_handle<Scalar>* gpu_ctx,
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta,
                    bool pin_host_buffers) {
    gpu::gemm(*(gpu_ctx), matrixA, matrixB, matrixC,
              m, n, k, alpha, beta,
              pin_host_buffers);
}
#endif

template <typename Scalar> 
Scalar& get_element(Scalar* mat, int m, int n, int i, int j) {
    return mat[j * m + i];
}

template <typename Scalar>
void local_multiply_cpu(
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta) {
    for (int mi = 0; mi < m; ++mi) {
        for (int ni = 0; ni < n; ++ni) {
            Scalar& Cvalue = get_element(matrixC, m, n, mi, ni);
            Cvalue *= beta;
            for (int ki = 0; ki < k; ++ki) {
                Scalar& Avalue = get_element(matrixA, m, k, mi, ki);
                Scalar& Bvalue = get_element(matrixB, k, n, ki, ni);
                Cvalue += alpha * Avalue * Bvalue;
            }
        }
    }
}

template <typename Scalar>
void local_multiply(cosma_context<Scalar>* ctx,
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta) {
    PE(multiply_computation);
#ifdef DEBUG
    auto t_start =
        debug_gemm_start(matrixA, matrixB, matrixC, m, n, k, alpha, beta);
#endif

#ifdef COSMA_HAVE_GPU
    if (ctx->pin_host_buffers) {
        ctx->get_memory_pool().pin(matrixA, m * k);
        ctx->get_memory_pool().pin(matrixB, k * n);
        ctx->get_memory_pool().pin(matrixC, m * n);
    }
    local_multiply(ctx->get_gpu_context(),
                   matrixA, matrixB, matrixC,
                   m, n, k, alpha, beta,
                   false);
#else
    gemm(m, n, k, alpha, matrixA, m, matrixB, k, beta, matrixC, m);
#endif
    PL();

#ifdef DEBUG
    auto t_end =
        debug_gemm_end(matrixA, matrixB, matrixC, m, n, k, alpha, beta);
    std::cout << "time(" << m << ", " << n << ", " << k
              << ") = " << std::chrono::duration_cast<ms_t>(t_end - t_start).count() << std::endl;
#endif
}

template <typename Scalar>
void local_multiply(Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta) {
    local_multiply(get_context_instance<Scalar>(), matrixA, matrixB, matrixC, m, n, k, alpha, beta);
}

template <typename Scalar>
void local_multiply(context<Scalar>& ctx,
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta) {
    local_multiply(ctx.get(), matrixA, matrixB, matrixC, m, n, k, alpha, beta);
}

// explicit template instantiation using context
template void local_multiply<double>(cosma_context<double> *ctx,
                                     double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double alpha,
                                     double beta);

template void local_multiply<float>(cosma_context<float> *ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta);

template void
local_multiply<std::complex<double>>(cosma_context<std::complex<double>> *ctx,
                                     std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta);

template void
local_multiply<std::complex<float>>(cosma_context<std::complex<float>> *ctx,
                                    std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta);

// explicit template instantiation using context - no pinning
template void local_multiply_cpu<double>(
                                     double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double alpha,
                                     double beta);

template void local_multiply_cpu<float>(
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta);

template void
local_multiply_cpu<std::complex<double>>(
                                     std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta);

template void
local_multiply_cpu<std::complex<float>>(
                                    std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta);

// explicit template instantiation using context with unique_ptr context
template void local_multiply<double>(context<double> &ctx,
                                     double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double alpha,
                                     double beta);

template void local_multiply<float>(context<float> &ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta);

template void
local_multiply<std::complex<double>>(context<std::complex<double>> &ctx,
                                     std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta);

template void
local_multiply<std::complex<float>>(context<std::complex<float>> &ctx,
                                    std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta);

// explicit instantiation without context
template void local_multiply<double>(double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double alpha,
                                     double beta);

template void local_multiply<float>(float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta);

template void
local_multiply<std::complex<double>>(std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta);

template void
local_multiply<std::complex<float>>(std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta);

#ifdef COSMA_HAVE_GPU
// explicit template instantiation using gpu context
template void local_multiply<double>(gpu::mm_handle<double> *ctx,
                                     double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double alpha,
                                     double beta,
                                     bool pin_host_buffers);

template void local_multiply<float>(gpu::mm_handle<float> *ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta,
                                    bool pin_host_buffers);

template void
local_multiply<std::complex<double>>(gpu::mm_handle<std::complex<double>> *ctx,
                                     std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta,
                                     bool pin_host_buffers);

template void
local_multiply<std::complex<float>>(gpu::mm_handle<std::complex<float>> *ctx,
                                    std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta,
                                    bool pin_host_buffers);
#endif
} // namespace cosma
