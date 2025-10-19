#include <cassert>
#include <complex>
#include <cosma/bfloat16.hpp>
#include <cosma/memory_pool.hpp>
#include <iostream>
#include <mpi.h>

template <typename T>
cosma::memory_pool<T>::memory_pool() {}

template <typename T>
cosma::memory_pool<T>::~memory_pool() {
    this->unpin_all();
}

template <typename T>
size_t cosma::memory_pool<T>::get_buffer_id(size_t size) {
    auto alignment = aligned_allocator<T>::get_alignment();
    assert(size > 0);

    // take the alignment into account
    if (alignment > 0) {
        size += aligned_allocator<T>::get_alignment_padding(size);
    }

    size_t offset = pool_size_;
    pool_size_ += size;
    ++n_buffers_;

    assert(alignment <= 0 ||
           aligned_allocator<T>::get_alignment_padding(offset) == 0);
    assert(alignment <= 0 ||
           aligned_allocator<T>::get_alignment_padding(pool_size_) == 0);
    return offset;
}

template <typename T>
T *cosma::memory_pool<T>::get_buffer_pointer(size_t id) {
    auto alignment = aligned_allocator<T>::get_alignment();
    assert(alignment <= 0 ||
           aligned_allocator<T>::get_alignment_padding(id) == 0);
    if (pool_size_ > pool_capacity_) {
        resize(pool_size_);
    }
    assert(id < pool_capacity_);
    return pool_.data() + id;
}

template <typename T>
void cosma::memory_pool<T>::free_buffer(T *ptr, size_t size) {
    auto alignment = aligned_allocator<T>::get_alignment();
    // take the alignment into account
    if (alignment > 0) {
        size += aligned_allocator<T>::get_alignment_padding(size);
        assert(aligned_allocator<T>::get_alignment_padding(size) == 0);
    }

    // std::cout << "freeing buffer of size " << size << ", current size =  " <<
    // pool_size_ << std::endl;
    assert(pool_size_ >= size);
    pool_size_ -= size;
    --n_buffers_;
    if (pool_.data() + pool_size_ != ptr) {
        std::cerr << "[COSMA][memory_pool] free mismatch size=" << size
                  << " expected=" << (void *)(pool_.data() + pool_size_)
                  << " got=" << (void *)ptr << std::endl;
    }
    // check if this buffer was on top of the memory pool
    assert(pool_.data() + pool_size_ == ptr);
    assert(alignment <= 0 ||
           aligned_allocator<T>::get_alignment_padding(pool_size_) == 0);
    // std::fill(ptr, ptr + size, T{});
}

template <typename T>
void cosma::memory_pool<T>::resize(size_t capacity) {
    auto alignment = aligned_allocator<T>::get_alignment();
    // resizing should always happen after reserve.
    // The reserve should take care that the reserved
    // memory is already aligned.
    assert(alignment <= 0 ||
           aligned_allocator<T>::get_alignment_padding(capacity) == 0);

    this->unpin_all();
    resized = true;
    already_pinned = false;
    try {
        pool_.resize(capacity);
    } catch (const std::bad_alloc &e) {
        std::cout
            << "COSMA (memory pool): not enough space. Try setting the CPU "
               "memory limit (see environment variable COSMA_CPU_MAX_MEMORY)."
            << std::endl;
        throw;
    } catch (const std::length_error &e) {
        std::cout
            << "COSMA (memory pool): size >= max_size(). Try setting the CPU "
               "memory limit (see environment variable COSMA_CPU_MAX_MEMORY)."
            << std::endl;
        throw;
    } catch (const std::exception &e) {
        std::cout << "COSMA (memory pool): unknown exception, potentially a "
                     "bug. Please inform us of the test-case."
                  << std::endl;
        throw;
    }
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
T *cosma::memory_pool<T>::get_pool_pointer() {
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
void cosma::memory_pool<T>::reserve(std::vector<size_t> &buffer_sizes) {
    auto alignment = aligned_allocator<T>::get_alignment();
    // total size of all buffers after aligning
    std::size_t size = 0;
    for (auto &buffer_size : buffer_sizes) {
        if (alignment > 0) {
            buffer_size +=
                aligned_allocator<T>::get_alignment_padding(buffer_size);
        }
        size += buffer_size;
    }

    // reserve a bit more for amortized resizing
    size = (std::size_t)std::ceil(size * amortization);
    // take the alignment into account
    if (alignment > 0) {
        size += aligned_allocator<T>::get_alignment_padding(size);
    }

    if (size > 0 && size > pool_capacity_) {
        pool_capacity_ = size;
        assert(alignment <= 0 || aligned_allocator<T>::get_alignment_padding(
                                     pool_capacity_) == 0);
        try {
            pool_.reserve(pool_capacity_);
        } catch (const std::bad_alloc &e) {
            std::cout << "COSMA (memory pool): not enough space. Try setting "
                         "the CPU memory limit (see environment variable "
                         "COSMA_CPU_MAX_MEMORY)."
                      << std::endl;
            throw;
        } catch (const std::length_error &e) {
            std::cout << "COSMA (memory pool): size >= max_size(). Try setting "
                         "the CPU memory limit (see environment variable "
                         "COSMA_CPU_MAX_MEMORY)."
                      << std::endl;
            throw;
        } catch (const std::exception &e) {
            std::cout << "COSMA (memory pool): unknown exception, potentially "
                         "a bug. Please inform us of the test-case."
                      << std::endl;
            throw;
        }
    }
}

template <typename T>
void cosma::memory_pool<T>::pin(T *ptr, std::size_t size) {
    auto alignment = aligned_allocator<T>::get_alignment();
    if (alignment > 0) {
        size += aligned_allocator<T>::get_alignment_padding(size);
    }
    // check if it's aligned
    assert(alignment <= 0 ||
           aligned_allocator<T>::get_alignment_padding(size) == 0);
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

#if defined(COSMA_WITH_GPU_AWARE_MPI) || defined(COSMA_WITH_NCCL)
template <typename T>
void cosma::memory_pool<T>::allocate_device_send_buffer(std::size_t size) {
    device_send_buffer.resize(size);
}

template <typename T>
void cosma::memory_pool<T>::allocate_device_receive_buffer(std::size_t size) {
    device_receive_buffer.resize(size);
}
#endif

template class cosma::memory_pool<double>;
template class cosma::memory_pool<float>;
template class cosma::memory_pool<std::complex<double>>;
template class cosma::memory_pool<std::complex<float>>;
template class cosma::memory_pool<cosma::bfloat16>;
