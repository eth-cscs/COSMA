#include "cosma/context.hpp"
#include <cosma/bfloat16.hpp>
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

// Specialized version for BF16 that uses FP32 accumulation
// This matches the behavior of MKL's cblas_gemm_bf16bf16f32
// (BF16×BF16→FP32→BF16) and prevents accumulation errors in the reference
// computation
template <>
void local_multiply_cpu<bfloat16>(bfloat16 *matrixA,
                                  bfloat16 *matrixB,
                                  bfloat16 *matrixC,
                                  int m,
                                  int n,
                                  int k,
                                  bfloat16 alpha,
                                  bfloat16 beta) {
    const float alpha_f = static_cast<float>(alpha);
    const float beta_f = static_cast<float>(beta);

    for (int mi = 0; mi < m; ++mi) {
        for (int ni = 0; ni < n; ++ni) {
            bfloat16 &Cvalue = get_element(matrixC, m, n, mi, ni);
            // Use FP32 accumulator for precision
            float acc = static_cast<float>(Cvalue) * beta_f;
            for (int ki = 0; ki < k; ++ki) {
                bfloat16 &Avalue = get_element(matrixA, m, k, mi, ki);
                bfloat16 &Bvalue = get_element(matrixB, k, n, ki, ni);
                acc += alpha_f * static_cast<float>(Avalue) *
                       static_cast<float>(Bvalue);
            }
            // Convert back to BF16
            Cvalue = bfloat16(acc);
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

// ============================================================================
// BFloat16 Specialization (Mixed Precision: BF16 × BF16 → FP32)
// ============================================================================
#if defined(COSMA_WITH_BLAS) || defined(COSMA_WITH_MKL_BLAS)

/**
 * @brief Specialized local multiply for BFloat16 with FP32 accumulation
 *
 * This specialization handles the mixed-precision case where inputs are BF16
 * but output and accumulation are in FP32. Note the signature differs from
 * the template: matrixC is float*, not bfloat16*.
 */
template <>
void local_multiply<bfloat16>(
    cosma_context<bfloat16> *ctx,
    bfloat16 *matrixA,
    bfloat16 *matrixB,
    bfloat16 *matrixC, // Actually unused, we write to FP32
    int m,
    int n,
    int k,
    bfloat16 alpha,
    bfloat16 beta,
    bool copy_c_back) {
    // For BF16, we need to handle mixed precision carefully
    // The gemm_bf16 function takes BF16 inputs but produces FP32 output
    // For now, we allocate a temporary FP32 buffer for the output

    // TODO: This is a workaround. Proper solution requires changing the
    // CosmaMatrix type system to support mixed-precision outputs.

    std::vector<float> C_fp32(m * n);

    // Convert alpha and beta to FP32
    float alpha_fp32 = static_cast<float>(alpha);
    float beta_fp32 = static_cast<float>(beta);

    // If beta != 0, we need to load existing C values (in FP32)
    if (std::abs(beta_fp32) > 0.0f) {
        for (int i = 0; i < m * n; ++i) {
            C_fp32[i] = static_cast<float>(matrixC[i]);
        }
    }

    PE(multiply_computation_gemm);
    gemm_bf16(m,
              n,
              k,
              alpha_fp32,
              matrixA,
              m,
              matrixB,
              k,
              beta_fp32,
              C_fp32.data(),
              m);
    PL();

    // Convert result back to BF16 (precision loss acceptable)
    if (copy_c_back) {
        for (int i = 0; i < m * n; ++i) {
            matrixC[i] = bfloat16(C_fp32[i]);
        }
    }
}

#endif // COSMA_WITH_BLAS || COSMA_WITH_MKL_BLAS

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

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

template void local_multiply_cpu<bfloat16>(bfloat16 *matrixA,
                                           bfloat16 *matrixB,
                                           bfloat16 *matrixC,
                                           int m,
                                           int n,
                                           int k,
                                           bfloat16 alpha,
                                           bfloat16 beta);

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

// BFloat16 instantiation (with context)
template void local_multiply<bfloat16>(context<bfloat16> &ctx,
                                       bfloat16 *matrixA,
                                       bfloat16 *matrixB,
                                       bfloat16 *matrixC,
                                       int m,
                                       int n,
                                       int k,
                                       bfloat16 alpha,
                                       bfloat16 beta,
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

#ifdef COSMA_GPU_HAS_BF16_SUPPORT
// explicit template instantiation for bfloat16 using gpu context
template void local_multiply<bfloat16>(gpu::mm_handle<bfloat16> *ctx,
                                       bfloat16 *matrixA,
                                       bfloat16 *matrixB,
                                       bfloat16 *matrixC,
                                       int m,
                                       int n,
                                       int k,
                                       bfloat16 alpha,
                                       bfloat16 beta,
                                       bool pin_host_buffers,
                                       bool copy_c_back);
#endif

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
