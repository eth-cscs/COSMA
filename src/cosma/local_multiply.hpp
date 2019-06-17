#pragma once

#include <cosma/blas.hpp>
#include <cosma/context.hpp>
#include <cosma/matrix.hpp>
#include <cosma/timer.hpp>

#include <semiprof.hpp>
#include <vector>
// #include <libsci_acc.h>

#ifdef COSMA_HAVE_GPU
#include <tiled_mm.hpp>
#endif

namespace cosma {
void local_multiply(context &ctx,
                    double *a,
                    double *b,
                    double *c,
                    int m,
                    int n,
                    int k,
                    double beta);
void local_multiply_cpu(double *a,
                        double *b,
                        double *c,
                        int m,
                        int n,
                        int k,
                        double beta);
} // namespace cosma
