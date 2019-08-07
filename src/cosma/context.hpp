#pragma once
#include <iostream>
#include <memory>

#ifdef COSMA_HAVE_GPU
#include <Tiled-MM/tiled_mm.hpp>
#endif

namespace cosma {

template <typename Scalar>
class context {
  public:
    context();
    context(int streams, int tile_m, int tile_n, int tile_k);

#ifdef COSMA_HAVE_GPU
    std::unique_ptr<gpu::mm_handle<Scalar>> gpu_ctx;
#endif
};

template <typename Scalar>
context<Scalar> make_context() {
    return context<Scalar>();
}

template <typename Scalar>
context<Scalar> make_context(int streams, int tile_m, int tile_n, int tile_k) {
    return context<Scalar>(streams, tile_m, tile_n, tile_k);
}

} // namespace cosma
