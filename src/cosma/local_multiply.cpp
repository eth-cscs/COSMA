#include <cosma/blas.hpp>
#include <cosma/local_multiply.hpp>
#include <cosma/timer.hpp>

#ifdef COSMA_HAVE_GPU
#include <tiled_mm.hpp>
#endif
#include <semiprof.hpp>

#include <chrono>
#include <vector>

namespace cosma {

using clock_t = std::chrono::high_resolution_clock;
using ms_t = std::chrono::milliseconds;

template <typename Scalar>
void print_matrix(int m, int n, Scalar *A, char label) {
    // TODO: Needs to be modified for complex numbers
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
clock_t::time_point debug_dgemm_start(Scalar *matrixA,
                                      Scalar *matrixB,
                                      Scalar *matrixC,
                                      int m,
                                      int n,
                                      int k,
                                      Scalar beta) {
    auto start = clock_t::now();
    if (beta > 0) {
        std::cout << "C (before) = " << std::endl;
        print_matrix(m, n, matrixC, 'C');
        auto C_partial = std::unique_ptr<Scalar[]>(new Scalar[m * n]);
        dgemm(m, n, k, 1.0, matrixA, m, matrixB, k, 0.0, C_partial.get(), m);
        std::cout << "C (partial) = " << std::endl;
        print_matrix(m, n, C_partial.get(), 'C');
    }
    return start;
}

template <typename Scalar>
clock_t::time_point debug_dgemm_end(Scalar *matrixA,
                                    Scalar *matrixB,
                                    Scalar *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    Scalar beta) {
    // TODO: generalie to complex numbers
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    print_matrix(m, k, matrixA, 'A');
    print_matrix(k, n, matrixB, 'B');
    print_matrix(m, n, matrixC, 'C');

    return std::chrono::high_resolution_clock::now();
}

template <typename Scalar>
void local_multiply(context &ctx,
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar beta) {
    PE(multiply_computation);
#ifdef DEBUG
    auto t_start = debug_dgemm_start(matrixA, matrixB, matrixC, m, n, k, beta);
#endif

#ifdef COSMA_HAVE_GPU
    gpu::dgemm(ctx.gpu_ctx, matrixA, matrixB, matrixC, m, n, k, 1.0, beta);
#else
    (void)ctx;
    dgemm(m, n, k, 1.0, matrixA, m, matrixB, k, beta, matrixC, m);
#endif
    // blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &beta,
    // matrixC, &m);
    PL();

#ifdef DEBUG
    auto t_end = debug_dgemm_end(matrixA, matrixB, matrixC, m, n, k, beta);
    std::cout << "time(" << m << ", " << n << ", " << k
              << ") = " << ms_t(t_end - t_start).count() << std::endl;
#endif
}

template <typename Scalar>
void local_multiply_cpu(Scalar *matrixA,
                        Scalar *matrixB,
                        Scalar *matrixC,
                        int m,
                        int n,
                        int k,
                        Scalar beta) {
#ifdef DEBUG
    auto t_start = debug_dgemm_start(matrixA, matrixB, matrixC, m, n, k, beta);
#endif
    PE(multiply_computation);
    dgemm(m, n, k, 1.0, matrixA, m, matrixB, k, beta, matrixC, m);
    PL();
#ifdef DEBUG
    auto t_end = debug_dgemm_end(matrixA, matrixB, matrixC, m, n, k, beta);
    std::cout << "time(" << m << ", " << n << ", " << k
              << ") = " << ms_t(t_end - t_start).count() << std::endl;
#endif
}

// TODO: complex instantiations;

template void local_multiply_cpu<double>(double *matrixA,
                                         double *matrixB,
                                         double *matrixC,
                                         int m,
                                         int n,
                                         int k,
                                         double beta);

template void local_multiply_cpu<float>(float *matrixA,
                                        float *matrixB,
                                        float *matrixC,
                                        int m,
                                        int n,
                                        int k,
                                        float beta);

template void local_multiply<double>(context &ctx,
                                     double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double beta);

template void local_multiply<float>(context &ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float beta);

} // namespace cosma
