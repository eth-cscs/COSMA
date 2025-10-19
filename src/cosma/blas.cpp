#include <cosma/blas.hpp>

#include <vector>

// extern "C" {
#ifdef COSMA_WITH_MKL_BLAS
#include <mkl.h>
#elif defined(COSMA_WITH_BLIS_BLAS)
#include <blis.h>
#elif defined(COSMA_WITH_BLAS)
#include <cblas.h>
// this is for backward compatibility,
// in case CBLAS_LAYOUT is not defined
typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif
// }

// The file is not needed if GPU is used
//
#if defined(COSMA_WITH_MKL_BLAS) || defined(COSMA_WITH_BLIS_BLAS) ||           \
    defined(COSMA_WITH_BLAS)
namespace cosma {
void gemm(const int M,
          const int N,
          const int K,
          const double alpha,
          const double *A,
          const int lda,
          const double *B,
          const int ldb,
          const double beta,
          double *C,
          const int ldc) {
    cblas_dgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

void gemm(const int M,
          const int N,
          const int K,
          const std::complex<double> alpha,
          const std::complex<double> *A,
          const int lda,
          const std::complex<double> *B,
          const int ldb,
          const std::complex<double> beta,
          std::complex<double> *C,
          const int ldc) {
    cblas_zgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                reinterpret_cast<const double *>(&alpha),
                reinterpret_cast<const double *>(A),
                lda,
                reinterpret_cast<const double *>(B),
                ldb,
                reinterpret_cast<const double *>(&beta),
                reinterpret_cast<double *>(C),
                ldc);
}

void gemm(const int M,
          const int N,
          const int K,
          const float alpha,
          const float *A,
          const int lda,
          const float *B,
          const int ldb,
          const float beta,
          float *C,
          const int ldc) {
    cblas_sgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

void gemm(const int M,
          const int N,
          const int K,
          const std::complex<float> alpha,
          const std::complex<float> *A,
          const int lda,
          const std::complex<float> *B,
          const int ldb,
          const std::complex<float> beta,
          std::complex<float> *C,
          const int ldc) {
    cblas_cgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                reinterpret_cast<const float *>(&alpha),
                reinterpret_cast<const float *>(A),
                lda,
                reinterpret_cast<const float *>(B),
                ldb,
                reinterpret_cast<const float *>(&beta),
                reinterpret_cast<float *>(C),
                ldc);
}

void gemm_bf16(const int M,
               const int N,
               const int K,
               const float alpha,
               const bfloat16 *A,
               const int lda,
               const bfloat16 *B,
               const int ldb,
               const float beta,
               float *C,
               const int ldc) {
#ifdef COSMA_WITH_MKL_BLAS
    // MKL 2020+ has native BF16 × BF16 → FP32 GEMM
    // Uses hardware-accelerated BF16 dot products on AVX-512 BF16 CPUs
    cblas_gemm_bf16bf16f32(CblasColMajor,
                           CblasNoTrans,
                           CblasNoTrans,
                           M,
                           N,
                           K,
                           alpha,
                           reinterpret_cast<const MKL_BF16 *>(A),
                           lda,
                           reinterpret_cast<const MKL_BF16 *>(B),
                           ldb,
                           beta,
                           C,
                           ldc);
#elif defined(COSMA_OPENBLAS_HAS_BF16_NATIVE)
    // OpenBLAS 0.3.27+ has native BF16 GEMM (cblas_sbgemm)
    // Uses AVX512_BF16 instructions when available
    // Note: OpenBLAS uses 'sbgemm' naming (single-precision BFloat16)
    // and outputs to FP32, matching the MKL behavior
    
    // OpenBLAS BF16 format: need to reinterpret bfloat16 as uint16_t storage
    // The actual cblas_sbgemm signature expects bfloat16 storage
    cblas_sbgemm(CblasColMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 M,
                 N,
                 K,
                 alpha,
                 reinterpret_cast<const bfloat16 *>(A),
                 lda,
                 reinterpret_cast<const bfloat16 *>(B),
                 ldb,
                 beta,
                 C,
                 ldc);
#else
    // Fallback: Convert BF16 → FP32, compute with FP32 GEMM
    // This is slower but works with any BLAS library

    // Allocate temporary FP32 buffers for A and B
    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(K * N);

    // Convert BF16 → FP32
    for (int i = 0; i < M * K; ++i) {
        A_fp32[i] = static_cast<float>(A[i]);
    }

    for (int i = 0; i < K * N; ++i) {
        B_fp32[i] = static_cast<float>(B[i]);
    }

    // Call standard FP32 GEMM
    cblas_sgemm(CBLAS_LAYOUT::CblasColMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                M,
                N,
                K,
                alpha,
                A_fp32.data(),
                lda,
                B_fp32.data(),
                ldb,
                beta,
                C,
                ldc);
#endif
}

// BF16 wrapper (converts output back to BF16)
void gemm(const int M,
          const int N,
          const int K,
          const bfloat16 alpha,
          const bfloat16 *A,
          const int lda,
          const bfloat16 *B,
          const int ldb,
          const bfloat16 beta,
          bfloat16 *C,
          const int ldc) {
    // Allocate FP32 buffer for output
    std::vector<float> C_fp32(M * N);

    // If beta != 0, convert existing C to FP32
    float beta_fp32 = static_cast<float>(beta);
    if (std::abs(beta_fp32) > 0.0f) {
        for (int i = 0; i < M * N; ++i) {
            C_fp32[i] = static_cast<float>(C[i]);
        }
    }

    // Call mixed-precision GEMM
    gemm_bf16(M,
              N,
              K,
              static_cast<float>(alpha),
              A,
              lda,
              B,
              ldb,
              beta_fp32,
              C_fp32.data(),
              ldc);

    // Convert output back to BF16
    for (int i = 0; i < M * N; ++i) {
        C[i] = bfloat16(C_fp32[i]);
    }
}

} // namespace cosma
#endif
