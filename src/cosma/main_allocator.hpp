#pragma once

#include <mpi.h>

#include <exception>
#include <iostream>
#include <limits>
#include <stdlib.h>
#if defined (__unix__)
#include <sys/mman.h>
#define ENABLE_HUGE_PAGES(ptr, size) \
{ \
    madvise(ptr, size, MADV_HUGEPAGE); \
}
#else
#define ENABLE_HUGE_PAGES(ptr, size)
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

  public:
    template <typename U>
    using rebind = main_allocator<U>;

  public:
    main_allocator() {}
    ~main_allocator() {}

    main_allocator(main_allocator const &) {}

    pointer address(reference r) { return &r; }

    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type cnt,
                     typename std::allocator<void>::const_pointer = 0) {
        if (cnt) {
            pointer ptr = static_cast<pointer>(
                              aligned_alloc(alignment, cnt * sizeof(T))
                          );

            ENABLE_HUGE_PAGES(ptr, cnt * sizeof(T));
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
