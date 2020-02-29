#ifdef COSMA_HAVE_GPU
#pragma once
#include <unordered_map>
#include <complex>

#include <Tiled-MM/util.hpp>

// container of pinned buffers
template <typename T>
struct pinned_buffers {
    std::unordered_map<T*, std::size_t> list;

    void add(T* ptr, std::size_t size);

    void clear();
};
#endif
