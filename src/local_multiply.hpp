#pragma once

#include <vector>
#include "blas.h"
#include "matrix.hpp"
#include <semiprof.hpp>
#include "timer.hpp"
#include "context.hpp"
// #include <libsci_acc.h>

#ifdef COSMA_HAVE_GPU
#include <tiled_mm.hpp>
#endif

namespace cosma {
void local_multiply(context& ctx, double* a, double* b, double* c, int m, int n, int k, double beta);
void local_multiply_cpu(double* a, double* b, double* c, int m, int n, int k, double beta);
}

