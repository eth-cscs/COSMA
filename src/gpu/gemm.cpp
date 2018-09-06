#ifdef COSMA_HAVE_GPU
#include "gemm.hpp"

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          double alpha, double beta) {

    // currently, running everything on GPU
    // variables needed for CPU+GPU overlap
    int m_gpu = m;
    int k_gpu = k;
    int n_gpu = n;

    int m_cpu = m;
    int k_cpu = k;
    int n_cpu = n - n_gpu;

    int lda = m;
    int ldb = k;
    int ldc = m;

    char N = 'N';
    double one = 1.;

    // copy to device
    cuda_stream stream; // default stream
    auto start_event = stream.enqueue_event();

    copy_to_device_async<value_type>(a, a_device, m_gpu*k_gpu, stream.stream());
    copy_to_device_async<value_type>(b, b_device, k_gpu*n_gpu, stream.stream());

    if (beta > 0) {
        copy_to_device_async<value_type>(c, c_device, m_gpu*n_gpu, stream.stream());
    }
    auto H2D_event = stream.enqueue_event();

    cublasDgemm(
            get_cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            m_gpu, n_gpu, k_gpu,
            &alpha,
            a_device, m_gpu,
            b_device, k_gpu,
            &beta,
            c_device, m_gpu);
    auto kernel_event = stream.enqueue_event();

    // copy result back to host
    copy_to_host_async<double>(c_device, c, m_gpu*n_gpu, stream.stream());
    auto end_event = stream.enqueue_event();

    // perform concurrently dgemm on CPU
    dgemm_(&N, &N, &m_cpu, &n_cpu, &k_cpu, &one, a, &lda, b + ldb * n_gpu, &ldb, &beta, c + ldc * n_gpu, &ldc);

    end_event.wait();

    //auto time_total = end_event.time_since(start_event);
    //auto time_H2D   = H2D_event.time_since(start_event);
    //auto time_D2H   = end_event.time_since(kernel_event);
    //auto time_dgemm = kernel_event.time_since(H2D_event);
}
#endif
