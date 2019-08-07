#pragma once
#include <vector>
#include <cosma/mpi_allocator.hpp>

namespace cosma {
template <typename T>
class memory_pool {
public:
    using mpi_buffer_t = std::vector<T, mpi_allocator<T>>;

    memory_pool() = default;
    memory_pool(size_t capacity);

    T* get_buffer(size_t size);
    void free_buffer(T* ptr);

    void resize(size_t capacity);
    void reset();

private:
    mpi_buffer_t pool_;
    size_t pool_size_;
    size_t pool_capacity_;
    size_t n_buffers_ = 0;
};
}
