#include "util.hpp"
#include "device_vector.hpp"

template <typename T>
device_vector<T>::device_vector(long long size) {
    ptr = malloc_device<T>(size);
}

template <typename T>
device_vector<T>::~device_vector() {
    cudaFree(ptr);
}

template <typename T>
T* device_vector<T>::data() {
    return ptr;
}

template device_vector<double>::device_vector(long long size);
template device_vector<double>::~device_vector();
template double* device_vector<double>::data();
