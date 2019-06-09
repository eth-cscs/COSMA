#include <cosma/local_multiply.hpp>

#include <chrono>

namespace cosma {
void print_matrix(int m, int n, double*A, char label) {
    std::cout << "Matrix " << label << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[j * m + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void local_multiply(context& ctx, double* matrixA, double* matrixB, double* matrixC,
        int m, int n, int k, double beta) {
    PE(multiply_computation);
    char N = 'N';
    double one = 1.;
#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    double zero = 0.;
    if (beta > 0) {
        std::cout << "C (before) = " << std::endl;
        print_matrix(m, n, matrixC, 'C');
        auto C_partial = std::unique_ptr<double[]>(new double[m * n]);
        blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &zero, C_partial.get(), &m);
        std::cout << "C (partial) = " << std::endl;
        print_matrix(m, n, C_partial.get(), 'C');
    }
#endif
#ifdef COSMA_HAVE_GPU
    // std::cout << "running from cpu gpu dgemm" << std::endl;
    gpu::dgemm(ctx.gpu_ctx, matrixA, matrixB, matrixC,
        m, n, k, 1.0, beta);
    // std::cout << "finished on cpu dgemm" << std::endl;
#else
    blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &beta, matrixC, &m);
    // dgemm_cpu('n', 'n', m, n, k, 1.0, matrixA, m, matrixB, k, beta, matrixC, m);
#endif
    // blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &beta, matrixC, &m);
    PL();
#ifdef DEBUG
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    print_matrix(m, k, matrixA, 'A');
    print_matrix(k, n, matrixB, 'B');
    print_matrix(m, n, matrixC, 'C');

    auto finish = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "time(" << m << ", " << n << ", " << k << ") = " << time <<  std::endl;
#endif
}

void local_multiply_cpu(double* matrixA, double* matrixB, double* matrixC,
        int m, int n, int k, double beta) {
    char N = 'N';
    double one = 1.;
#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    double zero = 0.;
    if (beta > 0) {
        std::cout << "C (before) = " << std::endl;
        print_matrix(m, n, matrixC, 'C');
        auto C_partial = std::unique_ptr<double[]>(new double[m * n]);
        blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &zero, C_partial.get(), &m);
        std::cout << "C (partial) = " << std::endl;
        print_matrix(m, n, C_partial.get(), 'C');
    }
#endif
    PE(multiply_computation);
    blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &beta, matrixC, &m);
    // blas::dgemm(&N, &N, &m, &n, &k, &one, matrixA, &m, matrixB, &k, &beta, matrixC, &m);
    PL();
#ifdef DEBUG
    std::cout << "After multiplication: " << std::endl;
    std::cout << "beta = " << beta << std::endl;
    print_matrix(m, k, matrixA, 'A');
    print_matrix(k, n, matrixB, 'B');
    print_matrix(m, n, matrixC, 'C');

    auto finish = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "time(" << m << ", " << n << ", " << k << ") = " << time <<  std::endl;
#endif
}
}
