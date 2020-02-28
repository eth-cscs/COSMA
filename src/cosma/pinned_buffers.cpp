#ifdef COSMA_HAVE_GPU
#include <cosma/pinned_buffers.hpp>

// container of pinned buffers
template <typename T>
void pinned_buffers<T>::add(T* ptr, std::size_t size) {
    // if already pinned, check if the same size
    auto elem_iter = list.find(ptr);
    if (elem_iter != list.end()) {
        if (elem_iter->second != size) {
            std::runtime_error("pinned_buffers.cpp: requested buffer \
                                already pinned with a different size.");
        }
    } else {
        // pin the buffer
        auto status = gpu::runtime_api::host_register(
                ptr,
                size * sizeof(T),
                gpu::runtime_api::flag::HostRegisterDefault);
        gpu::check_runtime_status(status);
        list[ptr] = size;
    }
}

template <typename T>
void pinned_buffers<T>::clear() {
    for (auto& elem : list) {
        // unpin the buffer
        auto status = gpu::runtime_api::host_unregister(elem.first);
        gpu::check_runtime_status(status);
    }
    list.clear();
}

// template instantiation for pinned_buffers
template struct pinned_buffers<float>;
template struct pinned_buffers<double>;
template struct pinned_buffers<std::complex<float>>;
template struct pinned_buffers<std::complex<double>>;
#endif
