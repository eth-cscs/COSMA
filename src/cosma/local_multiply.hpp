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
                    Scalar beta,
                    bool copy_c_back);

template <typename Scalar>
void local_multiply_cpu(
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
                    scalar beta,
                    bool copy_c_back);

template <typename scalar>
void local_multiply(scalar *a,
                    scalar *b,
                    scalar *c,
                    int m,
                    int n,
                    int k,
                    scalar alpha,
                    scalar beta,
                    bool copy_c_back);

#ifdef COSMA_HAVE_GPU
template <typename scalar>
void local_multiply(gpu::mm_handle<scalar>* gpu_ctx,
                    scalar *a,
                    scalar *b,
                    scalar *c,
                    int m,
                    int n,
                    int k,
                    scalar alpha,
                    scalar beta,
                    bool pin_host_buffers,
                    bool copy_c_back);
#endif
} // namespace cosma
