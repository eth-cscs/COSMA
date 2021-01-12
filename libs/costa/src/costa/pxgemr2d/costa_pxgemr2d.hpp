#pragma once
#include <complex>
#include <costa/scalapack.hpp>
/*
 * Redistributing the matrices
 */
namespace costa {

using zdouble_t = std::complex<double>;
using zfloat_t = std::complex<float>;

template <typename T>
void pxgemr2d(
           const int m,
           const int n,
           const T *a,
           const int ia,
           const int ja,
           const int *desca,
           T *b,
           const int ib,
           const int jb,
           const int *descb,
           const int ictxt);
} // namespace costa

