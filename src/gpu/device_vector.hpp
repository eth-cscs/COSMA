#pragma once
#include "util.hpp"

template <typename T>
class device_vector {
    T* data_ = nullptr;
    std::size_t size_ = 0lu;

    public:
    device_vector() = default;

    device_vector(std::size_t n): 
        data_(malloc_device<T>(n)),
        size_(n)
    {}

    T* data() {return data_;}
    std::size_t size() {return size_;}

    ~device_vector() {
        if (data_) cudaFree(data_);
    }
};

template device_vector<double>::device_vector(std::size_t);
template device_vector<double>::~device_vector();
template double* device_vector<double>::data();
template std::size_t device_vector<double>::size();
