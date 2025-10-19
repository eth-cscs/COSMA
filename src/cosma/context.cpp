#include <complex>
#include <stdlib.h>

#include "context.hpp"
#include <cosma/bfloat16.hpp>
#include <cosma/communicator.hpp>
#include <cosma/environment_variables.hpp>
#include <cosma/profiler.hpp>

namespace cosma {
#ifdef COSMA_HAVE_GPU
template <typename Scalar>
gpu::mm_handle<Scalar> *cosma_context<Scalar>::get_gpu_context() {
    return gpu_ctx_.get();
}
#endif
template <typename Scalar>
cosma_context<Scalar>::cosma_context() {
    cpu_memory_limit = get_cpu_max_memory<Scalar>();
    memory_pool_.amortization = get_memory_pool_amortization();
    adapt_to_scalapack_strategy = get_adapt_strategy();
    overlap_comm_and_comp = get_overlap_comm_and_comp();
    pin_host_buffers = get_memory_pinning();
#ifdef COSMA_HAVE_GPU
    gpu_ctx_ = gpu::make_context<Scalar>(
        gpu_streams(), gpu_max_tile_m(), gpu_max_tile_n(), gpu_max_tile_k());
#endif
}

template <typename Scalar>
cosma_context<Scalar>::cosma_context(size_t cpu_mem_limit,
                                     int streams,
                                     int tile_m,
                                     int tile_n,
                                     int tile_k) {
    cpu_memory_limit = (long long)cpu_mem_limit;
    adapt_to_scalapack_strategy = get_adapt_strategy();
    overlap_comm_and_comp = get_overlap_comm_and_comp();
    pin_host_buffers = get_memory_pinning();
    memory_pool_.amortization = get_memory_pool_amortization();
    // do not reserve nor resize the memory pool
    // let this just serve as the upper bound when creating a strategy
    // because otherwise, it might reserve/resize to much more than the problem
    // requires memory_pool_.resize(cpu_mem_limit);
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
    memory_pool_.unpin_all();
#ifdef DEBUG
    if (output) {
        std::cout << "context destroyed" << std::endl;
    }
#endif
}

template <typename Scalar>
memory_pool<Scalar> &cosma_context<Scalar>::get_memory_pool() {
    return memory_pool_;
}

template <typename Scalar>
long long cosma_context<Scalar>::get_cpu_memory_limit() {
    return cpu_memory_limit;
}

template <typename Scalar>
cosma::communicator *cosma_context<Scalar>::get_cosma_comm() {
    return prev_cosma_comm.get();
}

template <typename Scalar>
void cosma_context<Scalar>::register_state(MPI_Comm comm,
                                           const Strategy strategy) {
    if (comm == MPI_COMM_NULL)
        return;

    int same_comm = 0;

    if (!prev_cosma_comm || prev_cosma_comm->full_comm() == MPI_COMM_NULL) {
        prev_strategy = strategy;

        PE(preprocessing_communicators);
        prev_cosma_comm = std::make_unique<cosma::communicator>(strategy, comm);
        PL();
    } else {
        MPI_Comm prev_comm = prev_cosma_comm->full_comm();
        int comm_compare;
        MPI_Comm_compare(prev_comm, comm, &comm_compare);
        same_comm = comm_compare == MPI_CONGRUENT || comm_compare == MPI_IDENT;

        bool same_strategy = strategy == prev_strategy;

        // if same_comm and same strategy -> reuse the communicators
        if (!same_comm || !same_strategy) {
            prev_strategy = strategy;

            PE(preprocessing_communicators);
            prev_cosma_comm =
                std::make_unique<cosma::communicator>(strategy, comm);
            PL();

            memory_pool_.unpin_all();
            memory_pool_.already_pinned = false;
            memory_pool_.resized = false;
        }
    }

    // if this rank is not taking part in multiply, return
    // if (prev_cosma_comm->is_idle()) return;

#ifdef COSMA_HAVE_GPU
    if (!prev_cosma_comm->is_idle() && !memory_pool_.resized && same_comm &&
        strategy == prev_strategy) {
        memory_pool_.already_pinned = true;
    }
#endif
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
context<Scalar> make_context(size_t cpu_mem_limit,
                             int streams,
                             int tile_m,
                             int tile_n,
                             int tile_k) {
    return std::make_unique<cosma_context<Scalar>>(
        cpu_mem_limit, streams, tile_m, tile_n, tile_k);
}

// Meyer's singleton, thread-safe in C++11, but not in C++03.
// The thread-safety is guaranteed by the standard in C++11:
//     If control enters the declaration concurrently
//     while the variable is being initialized,
//     the concurrent execution shall wait
//     for completion of the initialization
template <typename Scalar>
global_context<Scalar> get_context_instance() {
    static context<Scalar> ctxt = make_context<Scalar>();
    return ctxt.get();
}

using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

// template instantiation for cosma_context
template class cosma_context<float>;
template class cosma_context<double>;
template class cosma_context<zfloat>;
template class cosma_context<zdouble>;
template class cosma_context<bfloat16>;

// template instantiation for make_context
template context<float> make_context();
template context<double> make_context();
template context<zfloat> make_context();
template context<zdouble> make_context();
template context<bfloat16> make_context();

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
template context<bfloat16> make_context(size_t cpu_mem_limit,
                                        int streams,
                                        int tile_m,
                                        int tile_n,
                                        int tile_k);

// template instantiation for get_context_instance
template global_context<float> get_context_instance();
template global_context<double> get_context_instance();
template global_context<zfloat> get_context_instance();
template global_context<zdouble> get_context_instance();
template global_context<bfloat16> get_context_instance();
} // namespace cosma
