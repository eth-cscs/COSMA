#pragma once

// mpi
#include <mpi.h>

// cosma
#include <cosma/communicator.hpp>
#include <cosma/context.hpp>
#include <cosma/interval.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>

// grid2grid
#include <transform.cpp>

namespace cosma {

/*
 * Performs matrix multiplication: C = alpha*A*B + beta*C,
 * where alpha and beta are scalars.
 */

/*
 * Takes matrices given in an arbitrary grid-like data layouts
 * grid2grid represents the abstract representation of the target layout
 * where target layout is the initial data layout for matrices A and B
 * and the final data layout for matrix C.
 * this function will perform the transformations between the target 
 * layouts and the optimal COSMA layout and perform the multiplication.
 * it is not as efficient as using the native COSMA layout,
 * but is very general as it can work with any grid-like data layout.
 */
template <typename Scalar>
void multiply_using_layout(
              grid2grid::grid_layout<Scalar> A_layout,
              grid2grid::grid_layout<Scalar> B_layout,
              grid2grid::grid_layout<Scalar> C_layout,
              int m, int n, int k,
              Scalar alpha, Scalar beta,
              MPI_Comm comm);

/*
 * Takes matrices in the optimal COSMA layout and the division strategy 
 * and performs the multiplication. It is very efficient as it uses the 
 * optimal COSMA layout.
 */
template <typename Scalar>
void multiply(context &ctx,
              CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              const Strategy &strategy,
              MPI_Comm comm,
              Scalar alpha,
              Scalar beta);

/*
 * Functions below are more interesting to the developer than the user.
 */
template <typename Scalar>
void multiply(context &ctx,
              CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              Interval &m,
              Interval &n,
              Interval &k,
              Interval &P,
              size_t step,
              const Strategy &strategy,
              communicator &comm,
              Scalar alpha,
              Scalar beta);

template <typename Scalar>
void sequential(context &ctx,
                CosmaMatrix<Scalar> &A,
                CosmaMatrix<Scalar> &B,
                CosmaMatrix<Scalar> &C,
                Interval &m,
                Interval &n,
                Interval &k,
                Interval &P,
                size_t step,
                const Strategy &strategy,
                communicator &comm,
                Scalar alpha,
                Scalar beta);

template <typename Scalar>
void parallel(context &ctx,
              CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              Interval &m,
              Interval &n,
              Interval &k,
              Interval &P,
              size_t step,
              const Strategy &strategy,
              communicator &comm,
              Scalar alpha,
              Scalar beta);

} // namespace cosma
