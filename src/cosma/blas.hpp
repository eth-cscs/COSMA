#pragma once
#include <mpi.h>

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
           const int ldc);

} // namespace cosma
