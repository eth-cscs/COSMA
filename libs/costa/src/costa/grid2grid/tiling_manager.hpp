#pragma once
#include <memory>

namespace costa {
namespace memory {

template <typename T>
struct tiling_manager {
    tiling_manager() = default;

    int block_dim = 512 / (int)sizeof(T);
    int max_threads = 2;
    std::unique_ptr<T[]> buffer = std::unique_ptr<T[]>(new T[block_dim * max_threads]);
};
}}
