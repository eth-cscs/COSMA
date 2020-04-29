#include <cassert>
#include <complex>
#include <cosma/memory_pool.hpp>
#include <iostream>
#include <mpi.h>

template <typename T>
cosma::memory_pool<T>::memory_pool() {
}

template <typename T>
cosma::memory_pool<T>::memory_pool(size_t capacity) {
    pool_.reserve(capacity);
}

template <typename T>
cosma::memory_pool<T>::~memory_pool() {
    this->unpin_all();
}

template <typename T>
size_t cosma::memory_pool<T>::get_buffer_id(size_t size) {
    assert(size > 0);
    size_t offset = pool_size_;
    pool_size_ += size;
    ++n_buffers_;
    return offset;
}

template <typename T>
T* cosma::memory_pool<T>::get_buffer_pointer(size_t id) {
    if (pool_size_ > pool_capacity_) {
        resize(pool_size_);
    }
    assert(id < pool_capacity_);
    return pool_.data() + id;
}

template <typename T>
void cosma::memory_pool<T>::free_buffer(T* ptr, size_t size) {
    // std::cout << "freeing buffer of size " << size << ", current size =  " << pool_size_ << std::endl;
    assert(pool_size_ >= size);
    pool_size_ -= size;
    --n_buffers_;
    // check if this buffer was on top of the memory pool
    assert(pool_.data() + pool_size_ == ptr);
    // std::fill(ptr, ptr + size, T{});
}

template <typename T>
void cosma::memory_pool<T>::resize(size_t capacity) {
    this->unpin_all();
    resized = true;
    already_pinned = false;
    pool_.resize(capacity);
    pool_size_ = capacity;
    pool_capacity_ = capacity;
}

template <typename T>
void cosma::memory_pool<T>::reset() {
    pool_size_ = 0;
    n_buffers_ = 0;
    this->unpin_all();
    resized = false;
    already_pinned = false;
}

template <typename T>
T* cosma::memory_pool<T>::get_pool_pointer() {
    return pool_.data();
}

template <typename T>
void cosma::memory_pool<T>::turn_on_output() {
    output = true;
}

template <typename T>
size_t cosma::memory_pool<T>::size() {
    return pool_size_;
}

template <typename T>
void cosma::memory_pool<T>::reserve(size_t size) {
    // reserve a bit more
    size += size / 10;
    if (size > 0 && size > pool_capacity_) {
        pool_capacity_ = size;
        pool_.reserve(pool_capacity_);
    }
}

template <typename T>
 void cosma::memory_pool<T>::reserve_additionally(size_t size) {
    // reserve a bit more
    size += size / 10;
     if (size > 0 && pool_size_ + size > pool_capacity_) {
         pool_capacity_ = pool_size_ + size;
         pool_.reserve(pool_capacity_);
     }
 }

template <typename T>
void cosma::memory_pool<T>::pin(T* ptr, std::size_t size) {
#ifdef COSMA_HAVE_GPU
    if (!already_pinned) {
        pinned_buffers_list.add(ptr, size);
    }
#endif
}

template <typename T>
void cosma::memory_pool<T>::unpin_all() {
#ifdef COSMA_HAVE_GPU
    pinned_buffers_list.clear();
#endif
}

template class cosma::memory_pool<double>;
template class cosma::memory_pool<float>;
template class cosma::memory_pool<std::complex<double>>;
template class cosma::memory_pool<std::complex<float>>;
