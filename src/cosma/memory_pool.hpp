#pragma once
#include <vector>
#include <cosma/pinned_buffers.hpp>

namespace cosma {
template <typename T>
class memory_pool {
public:
    using mpi_buffer_t = std::vector<T>;

    memory_pool();
    memory_pool(size_t capacity);

    ~memory_pool();

    // since vector can resize at some point,
    // we don't want to return pointer immediately
    // instead, when a new buffer is requested,
    // we return the current offset within the pool
    // that is used as an id of the buffer.
    // when the buffer actually used, its pointer
    // can be retrieved with get_buffer_pointer
    // that takes the buffer id (i.e. its offset within the pool)
    // and returns its pointer.
    size_t get_buffer_id(size_t size);
    T* get_buffer_pointer(size_t id);
    void free_buffer(T* ptr, size_t size);

    void resize(size_t capacity);
    void reset();

    T* get_pool_pointer();

    void turn_on_output();

    size_t size();
    void reserve(size_t size);
    void reserve_additionally(size_t size);

    void pin(T* ptr, std::size_t size);
    void unpin_all();

    // if true, buffers for this strategy are already pinned
    bool already_pinned = false;
    bool resized = false;

    // scaling factor for the buffer growth
    double amortization;

private:
    mpi_buffer_t pool_;
    size_t pool_size_ = 0;
    size_t pool_capacity_ = 0;
    size_t n_buffers_ = 0;
#ifdef COSMA_HAVE_GPU
    pinned_buffers<T> pinned_buffers_list;
#endif
    bool output = false;
};
}
