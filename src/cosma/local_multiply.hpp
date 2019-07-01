#pragma once

#include <cosma/context.hpp>

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
