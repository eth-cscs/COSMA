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
} // namespace cosma
