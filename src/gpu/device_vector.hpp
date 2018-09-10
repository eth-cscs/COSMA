#pragma once
#include "util.hpp"

template <typename T>
class device_vector {
public:
    device_vector() = default;

    device_vector(std::size_t n): 
        data_(malloc_device<T>(n)),
        size_(n) {}

    // copy-constructors are not supported
    device_vector(device_vector& other) = delete;
    // device_vector(device_vector&& other) = delete;

    // assignment operators are supported
    device_vector& operator=(device_vector&& other) {
        if (this != &other) {
            if (this->data_) {
                cudaFree(this->data_);
            }
            this->data_ = other.data_;
            other.data_ = nullptr;
            this->size_ = other.size_;
        }
        return *this;
    }

    T* data() {return data_;}
    std::size_t size() {return size_;}

    ~device_vector() {
        if (data_) {
            cudaFree(data_);
        }
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0lu;
};

template device_vector<double>::device_vector(std::size_t);
template device_vector<double>::~device_vector();
template double* device_vector<double>::data();
template std::size_t device_vector<double>::size();
