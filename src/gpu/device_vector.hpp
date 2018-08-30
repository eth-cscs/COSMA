#pragma once
// device vector
template <typename T>
class device_vector {
public:
    T* ptr;
    device_vector() = default;
    device_vector(long long size);

    ~device_vector();

    T* data();
};
