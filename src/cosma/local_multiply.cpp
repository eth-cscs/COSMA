#include "cosma/context.hpp"
#include <cosma/local_multiply.hpp>
#include <cosma/profiler.hpp>
#include <cosma/timer.hpp>

#ifdef COSMA_HAVE_GPU
#include <Tiled-MM/tiled_mm.hpp>
#include <Tiled-MM/util.hpp>

#ifdef COSMA_USE_UNIFIED_MEMORY
#include <Tiled-MM/gpu_blas_api.hpp>
#endif
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

#ifdef COSMA_USE_UNIFIED_MEMORY
using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

int get_first(char trans, int m, int n) { return trans == 'N' ? m : n; }

int get_second(char trans, int m, int n) { return trans == 'N' ? n : m; }

gpu::blas_api::OperationType get_blas_operation(char trans) {
    gpu::blas_api::OperationType op =
        trans == 'T'
            ? gpu::blas_api::operation::Transpose
            : (trans == 'C' ? gpu::blas_api::operation::ConjugateTranspose
                            : gpu::blas_api::operation::None);
    return op;
}

gpu::blas_api::StatusType cublas_gemm_wrapper(gpu::blas_api::HandleType handle,
                                              char trans_a,
                                              char trans_b,
                                              int m,
                                              int n,
                                              int k,
                                              const float *alpha,
                                              const float *a,
                                              const float *b,
                                              const float *beta,
                                              float *c,
                                              int lld_c) {
    gpu::blas_api::OperationType op_a = get_blas_operation(trans_a);
    gpu::blas_api::OperationType op_b = get_blas_operation(trans_b);

    int ld_a = get_first(trans_a, m, k);
    int ld_b = get_first(trans_b, k, n);

    return gpu::blas_api::sgemm(
        handle, op_a, op_b, m, n, k, alpha, a, ld_a, b, ld_b, beta, c, lld_c);
}

gpu::blas_api::StatusType cublas_gemm_wrapper(gpu::blas_api::HandleType handle,
                                              char trans_a,
                                              char trans_b,
                                              int m,
                                              int n,
                                              int k,
                                              const double *alpha,
                                              const double *a,
                                              const double *b,
                                              const double *beta,
                                              double *c,
                                              int lld_c) {
    gpu::blas_api::OperationType op_a = get_blas_operation(trans_a);
    gpu::blas_api::OperationType op_b = get_blas_operation(trans_b);

    int ld_a = get_first(trans_a, m, k);
    int ld_b = get_first(trans_b, k, n);

    return gpu::blas_api::dgemm(
        handle, op_a, op_b, m, n, k, alpha, a, ld_a, b, ld_b, beta, c, lld_c);
}

// Note: Converting from std::complex to cuComplex and cuDoubleComple
//       works because they are binary compatible.
//
//       http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=902
//
gpu::blas_api::StatusType cublas_gemm_wrapper(gpu::blas_api::HandleType handle,
                                              char trans_a,
                                              char trans_b,
                                              int m,
                                              int n,
                                              int k,
                                              const zfloat *alpha,
                                              const zfloat *a,
                                              const zfloat *b,
                                              const zfloat *beta,
                                              zfloat *c,
                                              int lld_c) {
    gpu::blas_api::OperationType op_a = get_blas_operation(trans_a);
    gpu::blas_api::OperationType op_b = get_blas_operation(trans_b);

    int ld_a = get_first(trans_a, m, k);
    int ld_b = get_first(trans_b, k, n);

    return gpu::blas_api::cgemm(
        handle,
        op_a,
        op_b,
        m,
        n,
        k,
        reinterpret_cast<const gpu::blas_api::ComplexFloatType *>(alpha),
        reinterpret_cast<const gpu::blas_api::ComplexFloatType *>(a),
        ld_a,
        reinterpret_cast<const gpu::blas_api::ComplexFloatType *>(b),
        ld_b,
        reinterpret_cast<const gpu::blas_api::ComplexFloatType *>(beta),
        reinterpret_cast<gpu::blas_api::ComplexFloatType *>(c),
        lld_c);
}

gpu::blas_api::StatusType cublas_gemm_wrapper(gpu::blas_api::HandleType handle,
                                              char trans_a,
                                              char trans_b,
                                              int m,
                                              int n,
                                              int k,
                                              const zdouble *alpha,
                                              const zdouble *a,
                                              const zdouble *b,
                                              const zdouble *beta,
                                              zdouble *c,
                                              int lld_c) {
    gpu::blas_api::OperationType op_a = get_blas_operation(trans_a);
    gpu::blas_api::OperationType op_b = get_blas_operation(trans_b);

    int ld_a = get_first(trans_a, m, k);
    int ld_b = get_first(trans_b, k, n);

    return gpu::blas_api::zgemm(
        handle,
        op_a,
        op_b,
        m,
        n,
        k,
        reinterpret_cast<const gpu::blas_api::ComplexDoubleType *>(alpha),
        reinterpret_cast<const gpu::blas_api::ComplexDoubleType *>(a),
        ld_a,
        reinterpret_cast<const gpu::blas_api::ComplexDoubleType *>(b),
        ld_b,
        reinterpret_cast<const gpu::blas_api::ComplexDoubleType *>(beta),
        reinterpret_cast<gpu::blas_api::ComplexDoubleType *>(c),
        lld_c);
}
#endif

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
void local_multiply(gpu::mm_handle<Scalar> *gpu_ctx,
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta,
                    bool pin_host_buffers,
                    bool copy_c_back) {
    /*
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        // print_matrix(m, k, matrixA, 'A');
        // print_matrix(k, n, matrixB, 'B');
        // std::cout << "m = " << m << ", n = " << n << ", k = " << k <<
    std::endl;
    }
    */
    int ld_a = m;
    int ld_b = k;
    int ld_c = m;

    gpu::gemm(*gpu_ctx,
              'N',
              'N',
              m,
              n,
              k,
              alpha,
              matrixA,
              ld_a,
              matrixB,
              ld_b,
              beta,
              matrixC,
              ld_c,
              pin_host_buffers,
              copy_c_back);

    /*
    if (rank == 0) {
        gpu::copy_to_host(gpu_ctx->get_full_device_buffer_c().data(), matrixC, m
    * n); print_matrix(m, n, matrixC, 'C'); std::cout << "alpha = " << alpha <<
    ", beta = " << beta << std::endl;
    }
    */
}
#endif

template <typename Scalar>
Scalar &get_element(Scalar *mat, int m, int n, int i, int j) {
    return mat[j * m + i];
}

template <typename Scalar>
void local_multiply_cpu(Scalar *matrixA,
                        Scalar *matrixB,
                        Scalar *matrixC,
                        int m,
                        int n,
                        int k,
                        Scalar alpha,
                        Scalar beta) {
    for (int mi = 0; mi < m; ++mi) {
        for (int ni = 0; ni < n; ++ni) {
            Scalar &Cvalue = get_element(matrixC, m, n, mi, ni);
            Cvalue *= beta;
            for (int ki = 0; ki < k; ++ki) {
                Scalar &Avalue = get_element(matrixA, m, k, mi, ki);
                Scalar &Bvalue = get_element(matrixB, k, n, ki, ni);
                Cvalue += alpha * Avalue * Bvalue;
            }
        }
    }
}

template <typename Scalar>
void local_multiply(cosma_context<Scalar> *ctx,
                    Scalar *matrixA,
                    Scalar *matrixB,
                    Scalar *matrixC,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta,
                    bool copy_c_back) {
#ifdef DEBUG
    auto t_start =
        debug_gemm_start(matrixA, matrixB, matrixC, m, n, k, alpha, beta);
#endif

#ifdef COSMA_HAVE_GPU
#ifdef COSMA_USE_UNIFIED_MEMORY
    if (ctx.unified_memory()) {
        PE(multiply_computation_gemm);
        auto status = cublas_gemm_wrapper(
            ctx->get_gpu_context()->get_gpu_context().get_blas_handle(0),
            'N',
            'N',
            m,
            n,
            k,
            &alpha,
            matrixA,
            matrixB,
            &beta,
            matrixC,
            m);

        gpu::check_blas_status(status);
        // we need explicit synchronization over the stream to trigger the copy
        // back to CPU memory
        hipStreamSynchronize(
            ctx->get_gpu_context()->get_gpu_context().get_stream(0));
        PL();
    } else {
#endif // COSMA_USE_UNIFIED_MEMORY
        PE(multiply_computation_pinning);
        if (ctx->pin_host_buffers) {
            ctx->get_memory_pool().pin(matrixA, m * k);
            ctx->get_memory_pool().pin(matrixB, k * n);
            // if (copy_c_back || std::abs(beta) > 0) {
            ctx->get_memory_pool().pin(matrixC, m * n);
            // }
        }
        PL();

        PE(multiply_computation_gemm);
        local_multiply(ctx->get_gpu_context(),
                       matrixA,
                       matrixB,
                       matrixC,
                       m,
                       n,
                       k,
                       alpha,
                       beta,
                       false,
                       copy_c_back);
        PL();
#ifdef COSMA_USE_UNIFIED_MEMORY
    }
#endif

#else
    PE(multiply_computation_gemm);
    gemm(m, n, k, alpha, matrixA, m, matrixB, k, beta, matrixC, m);
    PL();
#endif

#ifdef DEBUG
    auto t_end =
        debug_gemm_end(matrixA, matrixB, matrixC, m, n, k, alpha, beta);
    std::cout << "time(" << m << ", " << n << ", " << k << ") = "
              << std::chrono::duration_cast<ms_t>(t_end - t_start).count()
              << std::endl;
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
                    Scalar beta,
                    bool copy_c_back) {
    local_multiply(get_context_instance<Scalar>(),
                   matrixA,
                   matrixB,
                   matrixC,
                   m,
                   n,
                   k,
                   alpha,
                   beta,
                   copy_c_back);
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
                    Scalar beta,
                    bool copy_c_back) {
    local_multiply(ctx.get(),
                   matrixA,
                   matrixB,
                   matrixC,
                   m,
                   n,
                   k,
                   alpha,
                   beta,
                   copy_c_back);
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
                                     double beta,
                                     bool copy_c_back);

template void local_multiply<float>(cosma_context<float> *ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta,
                                    bool copy_c_back);

template void
local_multiply<std::complex<double>>(cosma_context<std::complex<double>> *ctx,
                                     std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta,
                                     bool copy_c_back);

template void
local_multiply<std::complex<float>>(cosma_context<std::complex<float>> *ctx,
                                    std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta,
                                    bool copy_c_back);

// explicit template instantiation using context - no pinning
template void local_multiply_cpu<double>(double *matrixA,
                                         double *matrixB,
                                         double *matrixC,
                                         int m,
                                         int n,
                                         int k,
                                         double alpha,
                                         double beta);

template void local_multiply_cpu<float>(float *matrixA,
                                        float *matrixB,
                                        float *matrixC,
                                        int m,
                                        int n,
                                        int k,
                                        float alpha,
                                        float beta);

template void
local_multiply_cpu<std::complex<double>>(std::complex<double> *matrixA,
                                         std::complex<double> *matrixB,
                                         std::complex<double> *matrixC,
                                         int m,
                                         int n,
                                         int k,
                                         std::complex<double> alpha,
                                         std::complex<double> beta);

template void
local_multiply_cpu<std::complex<float>>(std::complex<float> *matrixA,
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
                                     double beta,
                                     bool copy_c_back);

template void local_multiply<float>(context<float> &ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta,
                                    bool copy_c_back);

template void
local_multiply<std::complex<double>>(context<std::complex<double>> &ctx,
                                     std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta,
                                     bool copy_c_back);

template void
local_multiply<std::complex<float>>(context<std::complex<float>> &ctx,
                                    std::complex<float> *matrixA,
                                    std::complex<float> *matrixB,
                                    std::complex<float> *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    std::complex<float> alpha,
                                    std::complex<float> beta,
                                    bool copy_c_back);

// explicit instantiation without context
template void local_multiply<double>(double *matrixA,
                                     double *matrixB,
                                     double *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     double alpha,
                                     double beta,
                                     bool copy_c_back);

template void local_multiply<float>(float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta,
                                    bool copy_c_back);

template void
local_multiply<std::complex<double>>(std::complex<double> *matrixA,
                                     std::complex<double> *matrixB,
                                     std::complex<double> *matrixC,
                                     int m,
                                     int n,
                                     int k,
                                     std::complex<double> alpha,
                                     std::complex<double> beta,
                                     bool copy_c_back);

template void local_multiply<std::complex<float>>(std::complex<float> *matrixA,
                                                  std::complex<float> *matrixB,
                                                  std::complex<float> *matrixC,
                                                  int m,
                                                  int n,
                                                  int k,
                                                  std::complex<float> alpha,
                                                  std::complex<float> beta,
                                                  bool copy_c_back);

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
                                     bool pin_host_buffers,
                                     bool copy_c_back);

template void local_multiply<float>(gpu::mm_handle<float> *ctx,
                                    float *matrixA,
                                    float *matrixB,
                                    float *matrixC,
                                    int m,
                                    int n,
                                    int k,
                                    float alpha,
                                    float beta,
                                    bool pin_host_buffers,
                                    bool copy_c_back);

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
                                     bool pin_host_buffers,
                                     bool copy_c_back);

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
                                    bool pin_host_buffers,
                                    bool copy_c_back);
#endif
} // namespace cosma
