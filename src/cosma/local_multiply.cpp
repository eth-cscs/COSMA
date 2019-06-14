#include <cosma/local_multiply.hpp>

#include <chrono>

namespace cosma {

using clock_t = std::chrono::high_resolution_clock;
using ms_t = std::chrono::milliseconds;

void print_matrix(int m, int n, double *A, char label) {
    std::cout << "Matrix " << label << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[j * m + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

clock_t::time_point debug_dgemm_start(double *matrixA,
                                      double *matrixB,
                                      double *matrixC,
                                      int m,
                                      int n,
                                      int k,
                                      double beta) {
    auto start = clock_t::now();
    if (beta > 0) {
        std::cout << "C (before) = " << std::endl;
        print_matrix(m, n, matrixC, 'C');
        auto C_partial = std::unique_ptr<double[]>(new double[m * n]);
        dgemm(m, n, k, 1.0, matrixA, m, matrixB, k, 0.0, C_partial.get(), m);
        std::cout << "C (partial) = " << std::endl;
        print_matrix(m, n, C_partial.get(), 'C');
    }
    return start;
}

clock_t::time_point debug_dgemm_end(double *matrixA,
                                    double *matrixB,
                                    double *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    double beta) {
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    print_matrix(m, k, matrixA, 'A');
    print_matrix(k, n, matrixB, 'B');
    print_matrix(m, n, matrixC, 'C');

    return std::chrono::high_resolution_clock::now();
}

void local_multiply(context &ctx,
                    double *matrixA,
                    double *matrixB,
                    double *matrixC,
                    int m,
                    int n,
                    int k,
                    double beta) {
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

void local_multiply_cpu(double *matrixA,
                        double *matrixB,
                        double *matrixC,
                        int m,
                        int n,
                        int k,
                        double beta) {
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
} // namespace cosma
