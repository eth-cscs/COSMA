#pragma once
#include <complex>
#include <costa/scalapack.hpp>
/*
 * Transposing the matrices
 */
namespace costa {

using zdouble_t = std::complex<double>;
using zfloat_t = std::complex<float>;

template <typename T>
void pxtran(
           const int m,
           const int n,
           const T alpha,
           const T *a,
           const int ia,
           const int ja,
           const int *desca,
           const T beta,
           T *c,
           const int ic,
           const int jc,
           const int *descc);
} // namespace costa

