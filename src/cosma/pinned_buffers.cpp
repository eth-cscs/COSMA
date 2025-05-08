#include <complex>
#include <cosma/pinned_buffers.hpp>

// container of pinned buffers
template <typename T>
void pinned_buffers<T>::add(T* ptr, std::size_t size) {
    auto elem_iter = list.find(ptr);
    // if already pinned
    if (elem_iter != list.end()) {
        // check if the requested size is > pinned size
        // and in that case unpin the ptr
        if (size > elem_iter->second) {
            // unpin
            auto status = gpu::runtime_api::host_unregister((void*) ptr);
            gpu::check_runtime_status(status);

            // pin with the new size
            status = gpu::runtime_api::host_register(
                    (void*) ptr,
                    size * sizeof(T),
                    gpu::runtime_api::flag::HostRegisterDefault);
            gpu::check_runtime_status(status);
            elem_iter->second = size;
        }
    } else {
        // if not pinned previously
        // pin the buffer
        auto status = gpu::runtime_api::host_register(
                (void*) ptr,
                size * sizeof(T),
                gpu::runtime_api::flag::HostRegisterDefault);
        gpu::check_runtime_status(status);
        list.emplace(ptr, size);
    }
}

template <typename T>
void pinned_buffers<T>::clear() {
    for (auto& elem : list) {
        // unpin the buffer
        auto status = gpu::runtime_api::host_unregister((void*) elem.first);
        gpu::check_runtime_status(status);
    }
    list.clear();
}

// template instantiation for pinned_buffers
template struct pinned_buffers<float>;
template struct pinned_buffers<double>;
template struct pinned_buffers<std::complex<float>>;
template struct pinned_buffers<std::complex<double>>;
