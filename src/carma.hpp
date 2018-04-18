#pragma once

// STL
#include <vector>
#include "interval.hpp"
#include "communicator.hpp"
#include "blas.h"
#include "matrix.hpp"
#include <semiprof.hpp>
#include "strategy.hpp"
#include "topology.hpp"

void multiply(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
              const Strategy& strategy, MPI_Comm comm=MPI_COMM_WORLD);

void multiply(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
    Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
    const Strategy& strategy, double beta, MPI_Comm comm, MPI_Group comm_group);

void local_multiply(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
    int m, int n, int k, double beta);

void DFS(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
    Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
    const Strategy& strategy, double beta, MPI_Comm comm, MPI_Group comm_group);

void BFS(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
    Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
    const Strategy& strategy, double beta, MPI_Comm comm, MPI_Group comm_group);

// to achieve the maximum performance, blas should be invoked few times
// with a dummy computation, just so that it initializes the threading mechanism
void initialize_blas();
