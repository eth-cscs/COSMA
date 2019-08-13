#include <cosma/context.hpp>
#include <complex>
#include <limits>
#include <mpi.h>
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

// we want to cache cosma_context as an attribute to MPI_COMM_SELF
// so that it is destroyed when MPI_Finalize is invoked
//
int delete_fn(MPI_Datatype datatype, int key, void* attr_val, void * extra_state) {
    mpi_buffer_info attr = *(mpi_buffer_info*) (attr_val);
    if (attr.id == *(attr.global_counter) - 1 && attr.ptr) {
        MPI_Free_mem(attr.ptr);
    }
    return MPI_SUCCESS;
}

template <typename Scalar>
void cosma_context<Scalar>::register_to_destroy_at_finalize() {
    if (!mpi_keyval_set) {
        // keyval is global among all the ranks, but its value might differ among ranks
        // the value of this keyval is set with a call to MPI_Comm_set_attr
        MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, delete_fn, &mpi_keyval, NULL);
        mpi_keyval_set = true;
    }
    int old_counter = n_registers++;
    // old counter := the local counter that tells us in which call to this function
    //                the attribute was updated.
    // n_registers := the global counter that tells us how many times this function
    //                was invoked in total.
    // observe that this object takes a pointer to n_registers variable, 
    // so it "sees" the changes to n_registers variable 
    auto attr_val = mpi_buffer_info{memory_pool_.get_pool_pointer(), old_counter, &n_registers};
    buffer_infos.emplace_back(attr_val);
    // MPI_Comm_set_attr will invoke delete_fn if the attribute already exists. 
    // What we want here instead is to just update the value of the attribute, without previously 
    // invoking delete_fn, because our delete_fn function will deallocate all MPI buffers. 
    // We also want delete_fn to only deallocate buffers after the last time this function is called,
    // but we do not know in advance how many times this function will be invoked.
    // To achieve what we want, we added a condition inside the delete_fn, that checks whether
    // the local counter (old_counter) is equal to the global counter (n_registers).
    // We increase the value of the local counter before calling set, so that local counter is surely not equal
    // to the global one and thus delete_fn will do nothing after MPI_Comm_set_attr.
    // The delete_fn of this attribute will only deallocate the memory in the last call to this function
    // because only in that case local_counter == global_counter-1.
    // In our case, each call to multiply will call register_to_destroy_at_finalize, because that's when the
    // memory pool might have been resized, and for this reason we have to update the pointer to MPI buffers.
    ++n_registers;
    // the buffer_info object's lifetime has to be longer than the attribute lifetime,
    // so it has to be a member of this class.
    MPI_Comm_set_attr(MPI_COMM_SELF, mpi_keyval, &buffer_infos.back());
    // undo the artificial increase of n_registers, since MPI_Comm_set_attr
    // has already tried to invoke delete_fn (which we avoided), so we can undo.
    --n_registers;
}

template <typename Scalar>
void cosma_context<Scalar>::unregister_to_destroy_at_finalize() {
    // at this point we know that the memory pool is going out of scope
    // and that MPI_Finalize has not yet been invoked.
    // Thus we want to leave the deallocation of MPI memory to the 
    // destructor of memory_pool and not wait for MPI_Finalize.
    // We use a similar trick as in register_to_destroy_at_finalize,
    // to avoid delete_fn to act, we increase the global counter so that 
    // the condition inside the delete_fn is not fulfilled.
    ++n_registers;
}

template <typename Scalar>
cosma_context<Scalar>::~cosma_context() {
    unregister_to_destroy_at_finalize();
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
