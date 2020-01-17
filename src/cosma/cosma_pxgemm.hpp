#pragma once
#include <complex>
/*
 * This is a COSMA backend for matrices given in ScaLAPACK format.
 * It is less efficient than using cosma::multiply directly with COSMA data
 * layout. Thus, here we pay the price of transforming matrices between
 * scalapack and COSMA layout.
 */
namespace cosma {

using zdouble_t = std::complex<double>;
using zfloat_t = std::complex<float>;

template <typename T>
void pxgemm(const char trans_a,
           const char trans_b,
           const int m,
           const int n,
           const int k,
           const T alpha,
           const T *a,
           const int ia,
           const int ja,
           const int *desca,
           const T *b,
           const int ib,
           const int jb,
           const int *descb,
           const T beta,
           T *c,
           const int ic,
           const int jc,
           const int *descc);
} // namespace cosma
