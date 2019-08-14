#pragma once
#include <cosma/context.hpp>

namespace cosma {

template <typename Scalar>
void local_multiply(cosma_context<Scalar>* ctx,
                    Scalar *a,
                    Scalar *b,
                    Scalar *c,
                    int m,
                    int n,
                    int k,
                    Scalar alpha,
                    Scalar beta);

template <typename scalar>
void local_multiply(context<scalar>& ctx,
                    scalar *a,
                    scalar *b,
                    scalar *c,
                    int m,
                    int n,
                    int k,
                    scalar alpha,
                    scalar beta);

template <typename scalar>
void local_multiply(scalar *a,
                    scalar *b,
                    scalar *c,
                    int m,
                    int n,
                    int k,
                    scalar alpha,
                    scalar beta);
} // namespace cosma
