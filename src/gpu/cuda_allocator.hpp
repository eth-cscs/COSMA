#pragma once
#include "util.hpp"

/* 
 * A custom allocator that:
 *   - allocates pinned host memory using cudaHostAlloc
 */

template<typename T >
class cuda_allocator {
public:
    using value_type    = T;
    using pointer       = value_type*;
    using const_pointer = const value_type*;
    using reference     = value_type&;
    using const_reference = const value_type& ;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
public:
    template <typename U>
    using rebind = cuda_allocator<U>;
public:
    cuda_allocator() {}
    ~cuda_allocator() {}
    cuda_allocator(cuda_allocator const&) {}
    pointer address(reference r) {
        return &r;
    }
    const_pointer address(const_reference r) {
        return &r;
    }
    pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        if (cnt) {
            return malloc_pinned(cnt, value_type);
        }
        return nullptr;
    }
    void deallocate(pointer p, size_type cnt) {
        if (p) {
            cudaFreeHost(p);
        }
    }
    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
    void construct(pointer p, const T& t) {
        new(p) T(t);
    }
    void destroy(pointer p) {
        if (p) {
            p->~T();
        }
    }
    bool operator==(cuda_allocator const&) {
        return true;
    }
    bool operator!=(cuda_allocator const& a) {
        return !operator==(a);
    }
};
