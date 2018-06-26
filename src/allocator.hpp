#pragma once
#include <mpi.h>
#include <iostream>

template<typename T >
class allocator {
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
    using rebind = allocator<U>;
public:
    allocator() {}
    ~allocator() {}
    allocator(allocator const&) {}
    pointer address(reference r) {
        return &r;
    }
    const_pointer address(const_reference r) {
        return &r;
    }
    pointer allocate(size_type cnt, typename std::allocator<void>::const_pointer = 0) {
        if (cnt) {
            pointer ptr;
            MPI_Alloc_mem(cnt*sizeof(T), MPI_INFO_NULL, &ptr);
            return ptr;
        }
        return nullptr;
    }
    void deallocate(pointer p, size_type cnt) {
        if (p) {
            MPI_Free_mem(p);
        }
    }
    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
    void construct(pointer p, const T& t) {
        MPI_Alloc_mem(sizeof(T), MPI_INFO_NULL, &p);
        *p = T(t);
        //new(p) T(t);
    }
    void destroy(pointer p) {
        if (p) {
            p->~T();
            MPI_Free_mem(p);
        }
    }
    bool operator==(allocator const&) {
        return true;
    }
    bool operator!=(allocator const& a) {
        return !operator==(a);
    }
};
