#pragma once

#include <mpi.h>

#include <cassert>
#include <cosma/environment_variables.hpp>
#include <cosma/math_utils.hpp>
#include <exception>
#include <iostream>
#include <limits>

/*
 * A custom allocator that:
 *   - allocates the memory encouriging the use of huge pages
 *   - deallocates the memory
 */

namespace cosma {
template <typename T>
class aligned_allocator {
  public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // the alignement can be specified by the environment variable
    // or take its default value otherwise.
    // The default sizes, as well as the environment variable names
    // are defined in <cosma/environment_variables.hpp>
    static int get_alignment() {
        static int alignment = cosma::get_cosma_cpu_memory_alignment();
        return alignment;
    }

    // the minimum alignment for given type T
    std::size_t min_alignment() {
        return std::max(math_utils::next_power_of_2(sizeof(T)), sizeof(void *));
    }

    // Calculate how many additional elements we have to allocate for an array
    // of length n and data type T.
    static std::size_t get_alignment_padding(std::size_t n) {
        auto alignment = get_alignment();
        assert(alignment > 0);
        // Calculate the remainder in bytes (since the alignment is in bytes)
        auto remainder = (n * sizeof(T)) % alignment;

        // Convert the padding from bytes to the number of elements
        remainder = remainder != 0 ? (alignment - remainder) / sizeof(T) : 0;

        // std::cout << "For size " << n << ", reminder = " << remainder <<
        // std::endl; std::cout << "sizeof(T) = " << sizeof(T) << std::endl;
        return remainder;
    }

    // allocate memory with alignment specified as a template parameter
    // returns nullptr on failure
    T *aligned_malloc(std::size_t size) {
        auto alignment = get_alignment();
        // if alignment is disabled, use the standard malloc
        if (alignment <= 0) {
            return reinterpret_cast<T *>(malloc(size * sizeof(T)));
        }
        // check if the requested size is a multiple of the alignment
        assert(get_alignment_padding(size) == 0);
        // check if the alignment is >= min_alignment for this data type T
        assert(alignment >= min_alignment());
        // check if the alignment is a power of 2 and a multiple of
        // sizeof(void*).
        assert(math_utils::is_power_of_2(alignment));
        // "Memory alignment must be a power of 2.");
        // This is required for the posix_memalign function.
        assert(alignment % sizeof(void *) == 0);
        // "Memory alignment must be a multiple of sizeof(void*)");
        void *ptr;
        if (posix_memalign(&ptr, alignment, size * sizeof(T)) == 0) {
            return reinterpret_cast<T *>(ptr);
        }
        return nullptr;
    }

    aligned_allocator() {}
    ~aligned_allocator() {}

    aligned_allocator(aligned_allocator const &) {}

    pointer address(reference r) { return &r; }

    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type cnt,
                     typename std::allocator<void>::const_pointer = 0) {
        if (cnt > 0) {
            pointer ptr;
            if (!cosma::get_unified_memory()) {
                ptr = aligned_malloc(cnt);
#if defined(COSMA_USE_UNIFIED_MEMORY)
            } else {
                hipMalloc(&ptr, cnt * sizeof(T));
#else
            }
#endif
            }
            return ptr;
        }
        return nullptr;
    }

    void deallocate(pointer p, size_type cnt) {
        if (p) {
            if (!cosma::get_unified_memory())
                std::free(p);
#ifdef defined(COSMA_USE_UNIFIED_MEMORY)
            else
                hipFree(p);
#endif
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

    bool operator==(aligned_allocator const &) { return true; }

    bool operator!=(aligned_allocator const &a) { return !operator==(a); }
};

} // namespace cosma
