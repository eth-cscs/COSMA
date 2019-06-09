#pragma once
#include <iostream>
#include <memory>

#ifdef COSMA_HAVE_GPU
#include <tiled_mm.hpp>
#endif

namespace cosma {
class context {
public:
    context();
    context(int streams, int tile_m, int tile_n, int tile_k);

#ifdef COSMA_HAVE_GPU
    gpu::context gpu_ctx;
#endif
};
context make_context();
context make_context(int streams, int tile_m, int tile_n, int tile_k);
}
