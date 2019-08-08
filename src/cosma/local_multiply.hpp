#pragma once
#include <cosma/context.hpp>

namespace cosma {

template <typename Scalar>
void local_multiply(const context<Scalar> &ctx,
                    Scalar *a,
                    Scalar *b,
                    Scalar *c,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta);

} // namespace cosma
