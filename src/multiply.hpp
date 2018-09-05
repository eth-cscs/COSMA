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
#ifdef COSMA_HAVE_GPU
#include "gpu/gemm.hpp"
#endif

void multiply(CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
              const Strategy& strategy, MPI_Comm comm=MPI_COMM_WORLD, 
              bool one_sided_communication=false);

void multiply(CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
              Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
              const Strategy& strategy, double beta, communicator& comm);

void local_multiply(CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
                    int m, int n, int k, double beta);

void DFS(CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, double beta, communicator& comm);

void BFS(CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, double beta, communicator& comm);
