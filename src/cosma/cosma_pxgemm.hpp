#pragma once
#include <complex>
#include <cosma/scalapack.hpp>
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

/*
  If the matrix is very large, then its reshuffling is expensive.
  For this reason, try to adapt the strategy to the scalapack layout
  to minimize the need for reshuffling, even if it makes a 
  suoptimal communication scheme in COSMA.
*/
void adapt_strategy_to_block_cyclic_grid(// these will contain the suggested strategy prefix
                                         std::vector<int>& divisors, 
                                         std::string& dimensions,
                                         std::string& step_type,
                                         // multiplication problem size
                                         int m, int n, int k, int P,
                                         // global matrix dimensions
                                         scalapack::global_matrix_size& mat_dim_a,
                                         scalapack::global_matrix_size& mat_dim_b,
                                         scalapack::global_matrix_size& mat_dim_c,
                                         // block sizes
                                         scalapack::block_size& b_dim_a,
                                         scalapack::block_size& b_dim_b,
                                         scalapack::block_size& b_dim_c,
                                         // (i, j) denoting the submatrix coordinates
                                         int ia, int ja,
                                         int ib, int jb,
                                         int ic, int jc,
                                         // transpose flags
                                         char transa, char transb,
                                         // processor grid
                                         int procrows, int proccols,
                                         char order
                                         );
} // namespace cosma
