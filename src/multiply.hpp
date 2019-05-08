#pragma once

// STL
#include <vector>
#include "interval.hpp"
#include "communicator.hpp"
#include "hybrid_communicator.hpp"
#include "matrix.hpp"
#include <semiprof.hpp>
#include "strategy.hpp"
#include "timer.hpp"
#include "local_multiply.hpp"

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
