#include <cosma/context.hpp>
#include <complex>
#include <limits>
#include <cassert>

namespace cosma {
#ifdef COSMA_HAVE_GPU
template <typename Scalar>
gpu::mm_handle<Scalar>& cosma_context<Scalar>::get_gpu_ctx() {
    return gpu_ctx_;
}
#endif
template <typename Scalar>
cosma_context<Scalar>::cosma_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    memory_pool_.resize(cpu_mem_limit);
#ifdef COSMA_HAVE_GPU
    gpu_ctx = gpu::make_context<Scalar>(streams, tile_m, tile_n, tile_k);
#else
    std::cout << "Ignoring parameters in make_context. These parameters only "
                 "used in the CPU version."
              << std::endl;
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
context<Scalar> cosma::make_context() {
    auto ptr = std::make_unique<cosma_context<Scalar>>();
    return ptr.get();
}

template <typename Scalar>
context<Scalar> cosma::make_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    auto ptr = std::make_unique<cosma_context<Scalar>>(cpu_mem_limit, streams, tile_m, tile_n, tile_k);
    return ptr.get();
}

using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

// template instantiation for cosma_context
template class cosma::cosma_context<float>;
template class cosma::cosma_context<double>;
template class cosma::cosma_context<zfloat>;
template class cosma::cosma_context<zdouble>;

// template instantiation for make_context
template context<float> cosma::make_context();
template context<double> cosma::make_context();
template context<zfloat> cosma::make_context();
template context<zdouble> cosma::make_context();

template context<float> cosma::make_context(size_t cpu_mem_limit,
                                            int streams,
                                            int tile_m,
                                            int tile_n,
                                            int tile_k);
template context<double> cosma::make_context(size_t cpu_mem_limit,
                                             int streams,
                                             int tile_m,
                                             int tile_n,
                                             int tile_k);
template context<zfloat> cosma::make_context(size_t cpu_mem_limit,
                                             int streams,
                                             int tile_m,
                                             int tile_n,
                                             int tile_k);
template context<zdouble> cosma::make_context(size_t cpu_mem_limit,
                                              int streams,
                                              int tile_m,
                                              int tile_n,
                                              int tile_k);

// template instantiation for get_context_instance
template context<float> cosma::get_context_instance();
template context<double> cosma::get_context_instance();
template context<zfloat> cosma::get_context_instance();
template context<zdouble> cosma::get_context_instance();
} // namespace cosma
