#include <cosma/context.hpp>
#include <complex>
#include <limits>
#include <cassert>
#include <stdlib.h>
#include <limits>

namespace cosma {
int get_int_env_var(std::string name, int default_value) {
    char* var;
    var = getenv(name.c_str());
    int value = default_value;
    if (var != nullptr)
        value = std::atoi(var);
    return value;
}

int gpu_streams() {
    return get_int_env_var("COSMA_GPU_STREAMS", 2);
}

int gpu_max_tile_m() {
    return get_int_env_var("COSMA_GPU_MAX_TILE_M", 5000);
}

int gpu_max_tile_n() {
    return get_int_env_var("COSMA_GPU_MAX_TILE_N", 5000);
}

int gpu_max_tile_k() {
    return get_int_env_var("COSMA_GPU_MAX_TILE_K", 5000);
}

// reads the memory limit in MB per rank
// and converts the limit to #elements that each rank is allowed to use
template <typename T>
long long get_cpu_max_memory() {
    char* var;
    var = getenv ("COSMA_CPU_MAX_MEMORY");
    long long value = std::numeric_limits<long long>::max();
    long long megabytes = std::numeric_limits<long long>::max();
    if (var != nullptr) {
        megabytes = std::atoll(var);
        // from megabytes to #elements
        value = megabytes * 1024LL * 1024LL / sizeof(T);
    }

    return value;
}

bool env_var_defined(const char* var_name) {
    char* var = getenv (var_name);
    return var != nullptr;
}

#ifdef COSMA_HAVE_GPU
template <typename Scalar>
gpu::mm_handle<Scalar>* cosma_context<Scalar>::get_gpu_context() {
    return gpu_ctx_.get();
}
#endif
template <typename Scalar>
cosma_context<Scalar>::cosma_context() {
    cpu_memory_limit = get_cpu_max_memory<Scalar>();
#ifdef COSMA_HAVE_GPU
    gpu_ctx_ = gpu::make_context<Scalar>(gpu_streams(),
                                         gpu_max_tile_m(),
                                         gpu_max_tile_n(),
                                         gpu_max_tile_k());
#endif
}

template <typename Scalar>
cosma_context<Scalar>::cosma_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    cpu_memory_limit = (long long) cpu_mem_limit;
    // do not reserve nor resize the memory pool
    // let this just serve as the upper bound when creating a strategy
    // because otherwise, it might reserve/resize to much more than the problem requires
    // memory_pool_.resize(cpu_mem_limit);
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
memory_pool<Scalar>& cosma_context<Scalar>::get_memory_pool() {
    return memory_pool_;
}

template <typename Scalar>
long long cosma_context<Scalar>::get_cpu_memory_limit() {
    return cpu_memory_limit;
}

template <typename Scalar>
void cosma_context<Scalar>::register_state(int rank, const Strategy& strategy) {
#ifdef COSMA_HAVE_GPU
    if (memory_pool_.resized 
                || 
            rank != prev_rank
                ||
            strategy != prev_strategy
        ) {
        memory_pool_.unpin_all();
        memory_pool_.already_pinned = false;
        memory_pool_.resized = false;
        prev_rank = rank;
        prev_strategy = strategy;
    } else {
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
context<Scalar> make_context(size_t cpu_mem_limit, int streams, int tile_m, int tile_n, int tile_k) {
    return std::make_unique<cosma_context<Scalar>>(cpu_mem_limit, streams, tile_m, tile_n, tile_k);
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
