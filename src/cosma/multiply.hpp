#pragma once

#include <cosma/interval.hpp>
#include <cosma/communicator.hpp>
#include <cosma/hybrid_communicator.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <cosma/timer.hpp>
#include <cosma/local_multiply.hpp>

#include <semiprof.hpp>

#include <vector>

namespace cosma {
void multiply(context& ctx, CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
              const Strategy& strategy, MPI_Comm comm=MPI_COMM_WORLD,
              double beta = 0.0);

void multiply(context& ctx, CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
              Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
              const Strategy& strategy, communicator& comm, double beta = 0.0);

void sequential(context& ctx, CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, communicator& comm, double beta);

void parallel(context& ctx, CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, communicator& comm, double beta);

void parallel_overlapped(context& ctx, CosmaMatrix& A, CosmaMatrix& B, CosmaMatrix& C,
         Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
         const Strategy& strategy, communicator& comm, double beta);
}
