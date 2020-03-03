#pragma once

#include <mpi.h>

#include <exception>
#include <iostream>
#include <limits>
#include <stdlib.h>

#if defined (__unix__)
#include <sys/mman.h>
#define COSMA_HUGEPAGES_AVAILABLE
#endif

#if defined (__unix__) || (__APPLE__)
#define COSMA_POSIX_MEMALIGN_AVAILABLE
#endif

#if defined _ISOC11_SOURCE
#define COSMA_ALIGNED_ALLOC_AVAILABLE
#endif

/*
 * A custom allocator that:
 *   - allocates the memory encouriging the use of huge pages
 *   - deallocates the memory
 */

namespace cosma {
template <typename T>
class main_allocator {
public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // alignment corresponding to 2M page-size
    std::size_t alignment = (size_t) 2U * (size_t) 1024U * (size_t) 1024U;

    template <typename U>
    using rebind = main_allocator<U>;

    main_allocator() {}
    ~main_allocator() {}

    main_allocator(main_allocator const &) {}

    pointer address(reference r) { return &r; }

    const_pointer address(const_reference r) { return &r; }

    pointer allocate_aligned(size_type cnt) {
        void* ptr;
#ifdef COSMA_POSIX_MEMALIGN_AVAILABLE
        posix_memalign(&ptr, alignment, cnt * sizeof(value_type));
#elif COSMA_ALIGNED_ALLOC_AVAILABLE
        ptr = aligned_alloc(alignment, cnt * sizeof(value_type));
#else
        // don't align
        ptr = malloc(cnt * sizeof(value_type));
#endif
        return static_cast<pointer>(ptr);
    }

    void enable_huge_pages(pointer ptr, size_type cnt) {
#ifdef COSMA_HUGEPAGES_AVAILABLE
        madvise(ptr, cnt * sizeof(value_type), MADV_HUGEPAGE);
#endif
    }

    pointer allocate(size_type cnt,
                     typename std::allocator<void>::const_pointer = 0) {
        if (cnt) {
            pointer ptr = allocate_aligned(cnt);
            enable_huge_pages(ptr, cnt);
            return ptr;
        }
        return nullptr;
    }

    void deallocate(pointer p, size_type cnt) {
        if (p) {
            std::free(p);
        }
    }

    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    void construct(pointer p, const T &t) {
        new (p) T(t);
    }

    void destroy(pointer p) {
        if (p) {
            p->~T();
        }
    }

    bool operator==(main_allocator const &) { return true; }

    bool operator!=(main_allocator const &a) { return !operator==(a); }
};

} // namespace cosma
