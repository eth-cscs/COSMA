#pragma once
#include "bfloat16.hpp"
#include <complex>

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
          const int ldc);

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
          const int ldc);

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
          const int ldc);

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
          const int ldc);

/**
 * @brief Mixed-precision GEMM: BF16 × BF16 → FP32
 *
 * Performs C = alpha * A * B + beta * C where:
 * - A, B are in BFloat16 format (16-bit)
 * - C is in FP32 format (32-bit)
 * - Accumulation is done in FP32 for numerical accuracy
 *
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha FP32 scalar multiplier for A*B
 * @param A BF16 input matrix (M×K in column-major order)
 * @param lda Leading dimension of A (≥M)
 * @param B BF16 input matrix (K×N in column-major order)
 * @param ldb Leading dimension of B (≥K)
 * @param beta FP32 scalar multiplier for C
 * @param C FP32 output matrix (M×N in column-major order)
 * @param ldc Leading dimension of C (≥M)
 *
 * @note If MKL with BF16 support is available, uses cblas_gemm_bf16bf16f32.
 *       Otherwise, falls back to converting BF16→FP32, then using cblas_sgemm.
 */
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
               const int ldc);

/**
 * @brief BFloat16 GEMM wrapper (BF16 inputs and outputs)
 *
 * This is a convenience wrapper around gemm_bf16 that handles BF16 output.
 * Internally uses FP32 accumulation via gemm_bf16, then converts back to BF16.
 */
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
          const int ldc);

} // namespace cosma
