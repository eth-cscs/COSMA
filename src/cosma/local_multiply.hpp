#pragma once
#include <cosma/timer.hpp>
#include <cosma/context.hpp>
#include <cosma/blas.hpp>

#include <chrono>
#include <vector>
#include <semiprof.hpp>

#ifdef COSMA_HAVE_GPU
#include <tiled_mm.hpp>
#endif


namespace cosma {

template <typename Scalar>
void local_multiply(context &ctx,
                    Scalar *a,
                    Scalar *b,
                    Scalar *c,
                    int m,
                    int n,
                    int k,
                    Scalar beta);

template <typename Scalar>
void local_multiply_cpu(Scalar *a,
                        Scalar *b,
                        Scalar *c,
                        int m,
                        int n,
                        int k,
                        Scalar beta);
} // namespace cosma
