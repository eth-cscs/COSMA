#include <cosma/blas.hpp>

#ifdef COSMA_WITH_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace cosma {
void dgemm(const int M,
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

void dgemm(const int M,
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
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

void dgemm(const int M,
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

void dgemm(const int M,
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
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

} // namespace cosma
