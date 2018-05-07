#ifdef CARMA_HAVE_GPU
#include <iostream>
#include <cmath>
#include <cstdio>
#include "util.hpp"

using value_type = double;
using size_type  = size_t;

void gpu_dgemm_(double* a, double*b, double*c,
          int m, int n, int k,
          double alpha, double beta)
{

    double* a_device = malloc_device<value_type>(m*k);
    double* b_device = malloc_device<value_type>(k*n);
    double* c_device = malloc_device<value_type>(m*n);

    // copy to device
    cuda_stream stream; // default stream
    auto start_event = stream.enqueue_event();

    copy_to_device_async<value_type>(a, a_device, m*k, stream.stream());
    copy_to_device_async<value_type>(b, b_device, k*n, stream.stream());

    if (beta > 0) {
        copy_to_device_async<value_type>(c, c_device, m*n, stream.stream());
    }
    auto H2D_event = stream.enqueue_event();

    value_type alpha{1.};
    value_type beta{1.};

    cublasDgemm(
            get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            a_device, m,
            b_device, k,
            &beta,
            c_device, m);
    auto kernel_event = stream.enqueue_event();

    // copy result back to host
    copy_to_host_async<double>(c_device, c, m*n, stream.stream());
    auto end_event = stream.enqueue_event();
    end_event.wait();

    auto time_total = end_event.time_since(start_event);
    auto time_H2D   = H2D_event.time_since(start_event);
    auto time_D2H   = end_event.time_since(kernel_event);
    auto time_dgemm = kernel_event.time_since(H2D_event);

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
}
#endif
