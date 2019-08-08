#include <cosma/context.hpp>
#include <complex>
#include <limits>
#include <mpi.h>

namespace cosma {
#ifdef COSMA_HAVE_GPU
template <typename Scalar>
gpu::mm_handle<Scalar>& cosma_context<Scalar>::get_gpu_ctx() {
    return gpu_ctx_;
}
#endif

// constructor
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

// we want to cache cosma_context as an attribute to MPI_COMM_SELF
// so that it is destroyed when MPI_Finalize is invoked
//
int delete_fn(MPI_Datatype datatype, int key, void* attr_val, void * extra_state) {
    if (attr_val) {
        MPI_Free_mem(attr_val);
    }
    return MPI_SUCCESS;
}

template <typename Scalar>
void cosma_context<Scalar>::register_to_destroy_at_finalize() {
    if (!mpi_keyval_set) {
        MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, delete_fn, &mpi_keyval, NULL);
        mpi_keyval_set = true;
    }
    MPI_Comm_set_attr(MPI_COMM_SELF, mpi_keyval, memory_pool_.get_pool_pointer());
}

template <typename Scalar>
context<Scalar> cosma::make_context() {
    return std::make_unique<cosma_context<Scalar>>();
}

template <typename Scalar>
context<Scalar> cosma::make_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    return std::make_unique<cosma_context<Scalar>>(cpu_mem_limit, streams, tile_m, tile_n, tile_k);
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
template cosma_context<float>* const cosma::get_context_instance();
template cosma_context<double>* const cosma::get_context_instance();
template cosma_context<zfloat>* const cosma::get_context_instance();
template cosma_context<zdouble>* const cosma::get_context_instance();
} // namespace cosma
