#include <cassert>
#include <complex>
#include <cosma/memory_pool.hpp>

template <typename T>
cosma::memory_pool<T>::memory_pool(size_t capacity) {
    resize(capacity);
}

template <typename T>
size_t cosma::memory_pool<T>::get_buffer_id(size_t size) {
    assert(pool_size_ + size <= pool_capacity_);
    // T* ptr = pool_.data() + pool_size_;
    size_t offset = pool_size_;
    pool_size_ += size;
    ++n_buffers_;
    return offset;
}

template <typename T>
T* cosma::memory_pool<T>::get_buffer_pointer(size_t id) {
    assert(size < pool_capacity_);
    return pool_.data() + id;
}

template <typename T>
void cosma::memory_pool<T>::free_buffer(T* ptr, size_t size) {
    assert(pool_size_ >= size);
    pool_size_ -= size;
    --n_buffers_;
    // check if this buffer was on top of the memory pool
    assert(pool_.data() + pool_size_ == ptr);
}

template <typename T>
void cosma::memory_pool<T>::resize(size_t capacity) {
    pool_.resize(capacity);
    pool_capacity_ = capacity;
    pool_size_ = 0;
    n_buffers_ = 0;
}

template <typename T>
void cosma::memory_pool<T>::reset() {
    pool_size_ = 0;
    n_buffers_ = 0;
}

template <typename T>
T* cosma::memory_pool<T>::get_pool_pointer() {
    return pool_.data();
}

template class cosma::memory_pool<double>;
template class cosma::memory_pool<float>;
template class cosma::memory_pool<std::complex<double>>;
template class cosma::memory_pool<std::complex<float>>;
