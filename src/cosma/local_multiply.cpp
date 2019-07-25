#include <cosma/local_multiply.hpp>

#include <cosma/timer.hpp>
#ifdef COSMA_HAVE_GPU
#include <tiled_mm.hpp>
#endif
#ifdef COSMA_WITH_MKL
#include <cosma/blas.hpp>
#endif

#include <semiprof.hpp>

#include <chrono>
#include <complex>
#include <vector>

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

template <typename Scalar>
void local_multiply(context<Scalar> &ctx,
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
    gpu::gemm(*ctx.gpu_ctx, matrixA, matrixB, matrixC, m, n, k, alpha, beta);
#endif

#ifdef COSMA_WITH_MKL
    (void)ctx;
    gemm(m, n, k, alpha, matrixA, m, matrixB, k, beta, matrixC, m);
#endif
    // blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &beta,
    // matrixC, &m);
    PL();

#ifdef DEBUG
    auto t_end =
        debug_gemm_end(matrixA, matrixB, matrixC, m, n, k, alpha, beta);
    std::cout << "time(" << m << ", " << n << ", " << k
              << ") = " << ms_t(t_end - t_start).count() << std::endl;
#endif
}

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

} // namespace cosma
