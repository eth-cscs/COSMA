#pragma once

#include <mpi.h>

#include <exception>
#include <iostream>

/*
 * A custom allocator that:
 *   - allocates the memory using MPI_Alloc_mem and
 *   - deallocates the memory using MPI_Free_mem.
 *
 * Since it uses MPI routines, it has the following requirements:
 *   - it can only allocate the memory after either MPI_Init or MPI_Init_thread
 * are invoked
 *   - it can only deallocate the memory before MPI_Finalize is called.
 *
 * If any of these requirement is violated, the exception will be thrown.
 */

namespace cosma {
template <typename T>
class mpi_allocator {
  public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  public:
    template <typename U>
    using rebind = mpi_allocator<U>;

  public:
    mpi_allocator() {}
    ~mpi_allocator() {}

    mpi_allocator(mpi_allocator const &) {}

    pointer address(reference r) { return &r; }

    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type cnt,
                     typename std::allocator<void>::const_pointer = 0) {
        if (!mpi_enabled()) {
            throw_not_enabled_error();
            return nullptr;
        }
        if (cnt) {
            pointer ptr;
            MPI_Alloc_mem(cnt * sizeof(T), MPI_INFO_NULL, &ptr);
            return ptr;
        }
        return nullptr;
    }

    void deallocate(pointer p, size_type cnt) {
        if (!mpi_enabled()) {
            throw_not_enabled_error();
            return;
        }
        if (p) {
            MPI_Free_mem(p);
        }
    }

    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    void construct(pointer p, const T &t) { new (p) T(t); }

    void destroy(pointer p) {
        if (p) {
            p->~T();
        }
    }

    bool operator==(mpi_allocator const &) { return true; }

    bool operator!=(mpi_allocator const &a) { return !operator==(a); }

    bool mpi_enabled() {
        int initialized, finalized;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        return initialized && !finalized;
    }

    void throw_not_enabled_error() {
        std::runtime_error(
            "mpi_allocator must be constructed after MPI_Init/MPI_Init_thread \
                and destructed before MPI_Finalize is invoked. A typical mistake is to use \
                the mpi_allocator in the same scope with MPI_Init and MPI_Finalize. \
                In this case, the mpi_allocator will go out of the scope (and thus destructed) \
                only after MPI_Finalize. To prevent this, construct the objects that use \
                this allocator in a nested scope (or inside a new function).");
    }
};
} // namespace cosma
