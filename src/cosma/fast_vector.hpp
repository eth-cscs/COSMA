#pragma once
#include <complex>
#include <memory>
#include <mpi.h>

namespace cosma {
template <typename T>
class fast_vector {
public:
    using value_type = T;
    using pointer = value_type *;
    using size_type = std::size_t;

    // alignment corresponding to 2M page-size
    fast_vector() = default;

    // disable copy-constructor
    fast_vector(const fast_vector&) = delete;
    // disable copy-assignment
    fast_vector& operator=(fast_vector&) = delete;

    // move-constructor
    fast_vector(fast_vector&& other) {
        // assign values from other
        ptr_ = other.ptr_;
        size_ = other.size_;

        // discard all values in other
        other.ptr_ = nullptr;
        size_ = 0;
    }

    // move-assignment
    fast_vector& operator=(fast_vector&& other) {
        if (this != &other) {
            if (data() != nullptr) {
                free();
            }
            // assign values from other
            ptr_ = other.ptr_;
            size_ = other.size_;

            // discard all values in other
            other.ptr_ = nullptr;
            size_ = 0;
        }
        return *this;
    }

    ~fast_vector() {
        free();
    }

    void reserve(size_type new_size) {
        if (new_size > size_) {
            new_size = round_up(new_size);
            do_malloc(new_size);
        }
    }

    void resize(size_type new_size) {
        if (new_size > size_) {
            new_size = round_up(new_size);
            do_realloc(new_size);
        }
    }

    pointer data() {
        return ptr_;
    }

    size_type size() {
        return size_;
    }

private:
    void do_malloc(size_type cnt) {
        free();
        // cnt += cnt / 5;
        auto p = (void*) std::malloc(cnt * sizeof(value_type));
        if (p != nullptr) {
            size_ = cnt;
            ptr_ = static_cast<pointer>(p);
        } else {
            throw std::bad_alloc();
        }
    }

    void do_realloc(size_type cnt) {
        // cnt += cnt / 5;
        auto p = (void*) std::realloc(ptr_, cnt * sizeof(value_type));
        if (p != nullptr) {
            size_ = cnt;
            ptr_ = static_cast<pointer>(p);
        } else {
            throw std::bad_alloc();
        }
    }

    // grow with the factor of 1.2
    size_type round_up(size_type size) {
        return size + size/5;
    }

    void free() {
        // free the non-allocated pointer
        if (ptr_) {
            std::free(ptr_);
            size_ = 0;
            ptr_ = nullptr;
        }
    }

    pointer ptr_ = nullptr;
    size_type size_ = 0;
};
}
