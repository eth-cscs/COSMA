#include <cassert>
#include <complex>
#include <cosma/memory_pool.hpp>

template <typename T>
memory_pool<T>::memory_pool(size_t capacity) {
    resize(capacity);
}

template <typename T>
T* memory_pool<T>::get_buffer(size_t size) {
    assert(pool_size_ + size <= pool_capacity_);
    T* ptr = pool_.data() + pool_size_;
    pool_size_ += size;
    ++n_buffers_;
    return ptr;
}

template <typename T>
void free_buffer(T* ptr, size_t size) {
    assert(pool_size_ >= size);
    pool_size_ -= size;
    --n_buffers_;
    // check if this buffer was on top of the memory pool
    assert(pool_.data() + pool_size_ == ptr);
}

template <typename T>
void memory_pool<T>::resize(size_t capacity) {
    pool_.resize(capacity);
    pool_capacity_ = capacity;
    pool_size_ = 0;
    n_buffers_ = 0;
}

template <typename T>
void memory_pool<T>::reset() {
    pool_size_ = 0;
    n_buffers_ = 0;
}

template class memory_pool<double>;
template class memory_pool<float>;
template class memory_pool<std::complex<double>>;
template class memory_pool<std::complex<float>>;
