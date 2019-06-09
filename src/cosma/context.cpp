#include <cosma/context.hpp>

namespace cosma {
// constructor
context::context() {
#ifdef COSMA_HAVE_GPU
    gpu_ctx = gpu::make_context();
#endif
}

// constructor
context::context(int streams, int tile_m, int tile_n, int tile_k) {
#ifdef COSMA_HAVE_GPU
    gpu_ctx = gpu::make_context(streams, tile_m, tile_n, tile_k);
#else
    std::cout << "Ignoring parameters in make_context. These parameters only used in the CPU version." << std::endl;
#endif
}

context make_context() {
    return context();
}

context make_context(int streams, int tile_m, int tile_n, int tile_k) {
    return context(streams, tile_m, tile_n, tile_k);
}
}
