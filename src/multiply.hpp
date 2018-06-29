#pragma once

// STL
#include <vector>
#include "interval.hpp"
#include "communicator.hpp"
#include "one_sided_communicator.hpp"
#include "two_sided_communicator.hpp"
#include "blas.h"
#include "matrix.hpp"
#include <semiprof.hpp>
#include "strategy.hpp"
#include "timer.hpp"
#include "gpu/gemm.hpp"

void multiply(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
              const Strategy& strategy, MPI_Comm comm=MPI_COMM_WORLD, bool one_sided_communication=false);

void multiply(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
              Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
              const Strategy& strategy, double beta, communicator& comm);

void local_multiply(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
                    int m, int n, int k, double beta);

void DFS(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, double beta, communicator& comm);

void BFS(CarmaMatrix& A, CarmaMatrix& B, CarmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, double beta, communicator& comm);

// to achieve the maximum performance, blas should be invoked few times
// with a dummy computation, just so that it initializes the threading mechanism
void initialize_blas();
