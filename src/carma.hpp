#pragma once

// STL
#include <vector>
#include "interval.hpp"
#include "communicator.hpp"
#include "blas.h"
#include "matrix.hpp"
#include <omp.h>
#include <semiprof.hpp>
#include "strategy.hpp"

void multiply(CarmaMatrix *A, CarmaMatrix *B, CarmaMatrix *C,
              const Strategy& strategy);

void multiply(double *A, double *B, double *C,
    Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
    const Strategy& strategy, double beta, MPI_Comm comm);

void local_multiply(double *A, double *B, double *C,
    int m, int n, int k, double beta);

void DFS(double *A, double *B, double *C,
    Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
    const Strategy& strategy, double beta, MPI_Comm comm);

void BFS(double *A, double *B, double *C,
    Interval& m, Interval& n, Interval& k, Interval& P, size_t step,
    const Strategy& strategy, double beta, MPI_Comm comm);
