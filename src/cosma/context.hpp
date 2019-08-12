#pragma once
#include <iostream>
#include <memory>
#include <cosma/memory_pool.hpp>

#ifdef COSMA_HAVE_GPU
#include <Tiled-MM/tiled_mm.hpp>
#endif

namespace cosma {

template <typename Scalar>
class cosma_context {
public:
    cosma_context() = default;
    cosma_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k);

    memory_pool<Scalar>& get_memory_pool();
#ifdef COSMA_HAVE_GPU
    gpu::mm_handle<Scalar>& get_gpu_ctx();
#endif

    void register_to_destroy_at_finalize();

private:
    bool mpi_keyval_set = false;
    int mpi_keyval;
    memory_pool<Scalar> memory_pool_;
#ifdef COSMA_HAVE_GPU
    // std::unique_ptr<gpu::mm_handle<Scalar>> gpu_ctx;
    gpu::mm_handle<Scalar> gpu_ctx_;
#endif
};

template <typename Scalar>
using context = cosma_context<Scalar>*;

template <typename Scalar>
context<Scalar> make_context();

template <typename Scalar>
context<Scalar> make_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k);

// Meyer's singleton, thread-safe in C++11, but not in C++03.
// The thread-safety is guaranteed by the standard in C++11:
//     If control enters the declaration concurrently
//     while the variable is being initialized,
//     the concurrent execution shall wait
//     for completion of the initialization
template <typename Scalar>
static context<Scalar> get_context_instance() {
    static cosma_context<Scalar> ctxt;
    return &ctxt;
}

} // namespace cosma
