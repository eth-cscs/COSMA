#pragma once
#include <mpi.h>
#include <transform.cpp>

namespace cosma {
template <typename Scalar>
void multiply_using_layout(
              grid2grid::grid_layout<Scalar> A_layout,
              grid2grid::grid_layout<Scalar> B_layout,
              grid2grid::grid_layout<Scalar> C_layout,
              int m, int n, int k,
              Scalar alpha, Scalar beta,
              MPI_Comm comm);
}
