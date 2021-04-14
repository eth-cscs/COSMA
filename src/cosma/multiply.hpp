#pragma once

#include <cosma/communicator.hpp>
#include <cosma/context.hpp>
#include <cosma/interval.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>

#include <mpi.h>

#include <costa/grid2grid/transform.hpp>

namespace cosma {

/*
 * Performs matrix multiplication: C = alpha*op(A)*op(B) + beta*C,
 * where alpha and beta are scalars and op can be:
 * no-transpose ('N'), transpose ('T') or transpose and conjugate ('C')
 */

/*
 * Takes matrices given in an arbitrary grid-like data layouts
 * COSTA represents the abstract representation of the target layout
 * where target layout is the initial data layout for matrices A and B
 * and the final data layout for matrix C.
 * this function will perform the transformations between the target
 * layouts and the optimal COSMA layout and perform the multiplication.
 * it is not as efficient as using the native COSMA layout,
 * but is very general as it can work with any grid-like data layout.
 */
template <typename Scalar>
void multiply_using_layout(costa::grid_layout<Scalar> &A_layout,
                           costa::grid_layout<Scalar> &B_layout,
                           costa::grid_layout<Scalar> &C_layout,
                           Scalar alpha,
                           Scalar beta,
                           char transa,
                           char transb,
                           MPI_Comm comm);

/*
 * Takes matrices in the optimal COSMA layout and the division strategy
 * and performs the multiplication. It is very efficient as it uses the
 * optimal COSMA layout.
 */

template <typename Scalar>
void multiply(CosmaMatrix<Scalar> &A,
              CosmaMatrix<Scalar> &B,
              CosmaMatrix<Scalar> &C,
              const Strategy &strategy,
              MPI_Comm comm,
              Scalar alpha,
              Scalar beta);
} // namespace cosma
