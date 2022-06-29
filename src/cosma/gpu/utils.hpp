#pragma once

#include <cosma/gpu/gpu_runtime_api.hpp>

namespace cosma {
namespace gpu {
    void check_runtime_status(runtime_api::StatusType status) {
        if(status !=  runtime_api::status::Success) {
            std::cerr << "error: GPU API call : "
            << runtime_api::get_error_string(status) << std::endl;
            throw(std::runtime_error("GPU ERROR"));
        }
    }

    // copy n*T from host to device
    // If a cuda stream is passed as the final argument the copy will be performed
    // asynchronously in the specified stream, otherwise it will be serialized in
    // the default (NULL) stream
    template <typename T>
    void copy_to_device_async(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        auto status = runtime_api::memcpy_async(to, from, n * sizeof(T),
                runtime_api::flag::MemcpyHostToDevice, stream);
        check_runtime_status(status);
    }

    // copy n*T from device to host
    // If a cuda stream is passed as the final argument the copy will be performed
    // asynchronously in the specified stream, otherwise it will be serialized in
    // the default (NULL) stream
    template <typename T>
    void copy_to_host_async(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        auto status = runtime_api::memcpy_async(to, from, n * sizeof(T),
                                                runtime_api::flag::MemcpyDeviceToHost, stream);
        check_runtime_status(status);
    }

    template <typename T>
    void copy_device_to_device_async(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        auto status = runtime_api::memcpy_async(to, from, n * sizeof(T),
                                                runtime_api::flag::MemcpyDeviceToDevice, stream);
        check_runtime_status(status);
    }


} // gpu
} // cosma
