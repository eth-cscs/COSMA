#include <cosma/context.hpp>
#include <complex>
#include <limits>
#include <cassert>
#include <stdlib.h>

namespace cosma {

int get_num_ranks_per_gpu() {
    char* var;
    var = getenv ("COSMA_RANKS_PER_GPU");
    int ranks_per_gpu = 1;
    if (var != nullptr)
        ranks_per_gpu = std::atoi(var);
    return ranks_per_gpu;
}

#ifdef COSMA_HAVE_GPU
template <typename Scalar>
gpu::mm_handle<Scalar>* cosma_context<Scalar>::get_gpu_context() {
    return gpu_ctx_.get();
}
#endif
template <typename Scalar>
cosma_context<Scalar>::cosma_context() {
#ifdef COSMA_HAVE_GPU
    gpu_ctx_ = gpu::make_context<Scalar>(get_num_ranks_per_gpu());
#endif
}

template <typename Scalar>
cosma_context<Scalar>::cosma_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    memory_pool_.resize(cpu_mem_limit);
#ifdef COSMA_HAVE_GPU
    gpu_ctx_ = gpu::make_context<Scalar>(streams, tile_m, tile_n, tile_k);
#else
    std::cout << "Ignoring parameters in make_context. These parameters only "
                 "used in the CPU version."
              << std::endl;
#endif
}

template <typename Scalar>
cosma_context<Scalar>::~cosma_context() {
#ifdef DEBUG
    if (output) {
        std::cout << "context destroyed" << std::endl;
    }
#endif
}

template <typename Scalar>
memory_pool<Scalar>& cosma_context<Scalar>::get_memory_pool() {
    return memory_pool_;
}


template <typename Scalar>
void cosma_context<Scalar>::register_to_destroy_at_finalize() {
    // This function is called with each multiplication, because that's when 
    // the memory pool might have been resized. This function updates the value of 
    // the MPI buffer pointer (from the memory pool) through an attribute associated 
    // with MPI_COMM_SELF, so that MPI_Finalize can deallocate the MPI buffer 
    // (through attribute destruction function that we provide: delete_fn) in case
    // no context destructor was invoked before MPI_Finalize.
    attr.update_attribute(memory_pool_.get_pool_pointer());
}

template <typename Scalar>
void cosma_context<Scalar>::turn_on_output() {
    output = true;
    memory_pool_.turn_on_output();
}

template <typename Scalar>
context<Scalar> make_context() {
    return std::make_unique<cosma_context<Scalar>>();
}

template <typename Scalar>
context<Scalar> make_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    return std::make_unique<cosma_context<Scalar>>(cpu_mem_limit, streams, tile_m, tile_n, tile_k);
}

using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

// template instantiation for cosma_context
template class cosma_context<float>;
template class cosma_context<double>;
template class cosma_context<zfloat>;
template class cosma_context<zdouble>;

// template instantiation for make_context
template context<float> make_context();
template context<double> make_context();
template context<zfloat> make_context();
template context<zdouble> make_context();

template context<float> make_context(size_t cpu_mem_limit,
                                            int streams,
                                            int tile_m,
                                            int tile_n,
                                            int tile_k);
template context<double> make_context(size_t cpu_mem_limit,
                                             int streams,
                                             int tile_m,
                                             int tile_n,
                                             int tile_k);
template context<zfloat> make_context(size_t cpu_mem_limit,
                                             int streams,
                                             int tile_m,
                                             int tile_n,
                                             int tile_k);
template context<zdouble> make_context(size_t cpu_mem_limit,
                                              int streams,
                                              int tile_m,
                                              int tile_n,
                                              int tile_k);

// template instantiation for get_context_instance
template global_context<float> get_context_instance();
template global_context<double> get_context_instance();
template global_context<zfloat> get_context_instance();
template global_context<zdouble> get_context_instance();
}
