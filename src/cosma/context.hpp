#pragma once
#include <iostream>
#include <memory>
#include <cosma/memory_pool.hpp>
#include <cosma/strategy.hpp>
// #include <cosma/communicator.hpp>

#include <mpi.h>

#ifdef COSMA_HAVE_GPU
#include <Tiled-MM/tiled_mm.hpp>
#endif

namespace cosma {

// forward-declaration
class communicator;

template <typename Scalar>
class cosma_context {
public:
    cosma_context();
    cosma_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k);
    ~cosma_context();

    void register_state(MPI_Comm comm, 
                        const Strategy& strategy);

    memory_pool<Scalar>& get_memory_pool();
#ifdef COSMA_HAVE_GPU
    gpu::mm_handle<Scalar>* get_gpu_context();
#endif

    cosma::communicator* get_cosma_comm();

    long long get_cpu_memory_limit();

    void turn_on_output();

    bool adapt_to_scalapack_strategy = true;

    bool overlap_comm_and_comp = false;

    bool pin_host_buffers = true;

private:
    long long cpu_memory_limit = std::numeric_limits<long long>::max();
    memory_pool<Scalar> memory_pool_;
#ifdef COSMA_HAVE_GPU
    std::unique_ptr<gpu::mm_handle<Scalar>> gpu_ctx_;
    // gpu::mm_handle<Scalar> gpu_ctx_;
#endif
    bool output = false;
    Strategy prev_strategy;
    std::unique_ptr<cosma::communicator> prev_cosma_comm;
};

template <typename Scalar>
using global_context = cosma_context<Scalar>*;

template <typename Scalar>
using context = std::unique_ptr<cosma_context<Scalar>>;

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
global_context<Scalar> get_context_instance();
} // namespace cosma
