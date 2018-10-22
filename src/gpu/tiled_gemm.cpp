#include "gemm.hpp"

int actual_size(int n_tiles, int tile_id, int tile_length, int tile_remainder) {
    bool last_tile = tile_id == n_tiles - 1;
    bool not_divisible = tile_remainder > 0;

    return last_tile && not_divisible ? tile_remainder : tile_length;
}

void copy_tile_to_device_async(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id,
        cuda_stream& stream) {

    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    int tile_size = tile_m * tile_n;
    int offset_to = stream_id * tile_size;

    int tile_offset_global = (p_col_tile*tile_n) * m;
    int el_offset_global = p_row_tile * tile_m;
    int offset_from = tile_offset_global + el_offset_global;

    cudaMemcpy2DAsync(to + offset_to, actual_tile_m * sizeof(double),
            from + offset_from, m * sizeof(double),
            actual_tile_m * sizeof(double), actual_tile_n,
            cudaMemcpyHostToDevice, stream.stream());
}

void copy_tile_to_device(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id) {
    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    int tile_size = tile_m * tile_n;
    int offset_to = stream_id * tile_size;

    int tile_offset_global = (p_col_tile*tile_n) * m;
    int el_offset_global = p_row_tile * tile_m;
    int offset_from = tile_offset_global + el_offset_global;

    cudaMemcpy2D(to + offset_to, actual_tile_m * sizeof(double),
            from + offset_from, m * sizeof(double),
            actual_tile_m * sizeof(double), actual_tile_n,
            cudaMemcpyHostToDevice);
}

void copy_tile_to_host_async(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id,
        cuda_stream& stream) {
    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    int tile_size = tile_m * tile_n;
    int offset_from = stream_id * tile_size;

    int tile_offset_global = (p_col_tile*tile_n) * m;
    int el_offset_global = p_row_tile * tile_m;
    int offset_to = tile_offset_global + el_offset_global;

    cudaMemcpy2DAsync(to + offset_to, m * sizeof(double),
            from + offset_from, actual_tile_m * sizeof(double),
            actual_tile_m * sizeof(double), actual_tile_n,
            cudaMemcpyDeviceToHost, stream.stream());
}

void copy_tile_to_host(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id) {
    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    int tile_size = tile_m * tile_n;
    int offset_from = stream_id * tile_size;

    int tile_offset_global = (p_col_tile*tile_n) * m;
    int el_offset_global = p_row_tile * tile_m;
    int offset_to = tile_offset_global + el_offset_global;

    cudaMemcpy2D(to + offset_to, m * sizeof(double),
            from + offset_from, actual_tile_m * sizeof(double),
            actual_tile_m * sizeof(double), actual_tile_n,
            cudaMemcpyDeviceToHost);
}

void gpu_dgemm_(double* a, double* b, double* c,
        double* a_device, double* b_device, double* c_device,
        int m, int n, int k,
        int tile_size_m, int tile_size_n, int tile_size_k,
        double alpha, double beta) {

    tile_size_m = std::min(tile_size_m, m);
    tile_size_n = std::min(tile_size_n, n);
    tile_size_k = std::min(tile_size_k, k);

    // short tile sizes (when dim not divisible by tile)
    int short_tile_size_m = m % tile_size_m;
    int short_tile_size_n = n % tile_size_n;
    int short_tile_size_k = k % tile_size_k;

    int offset_a = tile_size_m * tile_size_k;
    int offset_b = tile_size_k * tile_size_n;
    int offset_c = tile_size_m * tile_size_n;

    int n_tiles_m = (int) std::ceil(1.0 * m / tile_size_m);
    int n_tiles_n = (int) std::ceil(1.0 * n / tile_size_n);
    int n_tiles_k = (int) std::ceil(1.0 * k / tile_size_k);

    // if the actual number of tiles is smaller 
    // than the number of streams => use less streams
    int n_streams = std::min(N_STREAMS, n_tiles_m * n_tiles_n * n_tiles_k);

    std::vector<cuda_stream> streams(n_streams);

#pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < n_streams; ++i) {
        // let each thread use a separate cuBLAS handle
        // and let each handle be bound to a separate stream
        cublasSetStream(get_cublas_handle(i), streams[i].stream());
    }

    // contains the last events on each stream
    std::vector<cuda_event> last_event_on_stream(n_streams);

    cuda_event copy_event;

// ***********************
//    MULTIPLICATION
// ***********************
#pragma omp parallel for collapse(2) num_threads(n_streams) shared (copy_event) schedule(dynamic)
    // loop over row tiles
    for (int m_tile_id = 0; m_tile_id < n_tiles_m; m_tile_id++) {
        // loop over column tiles
        for (int n_tile_id = 0; n_tile_id < n_tiles_n; n_tile_id++) {
            int actual_size_m = actual_size(n_tiles_m, m_tile_id, tile_size_m, short_tile_size_m);
            int actual_size_n = actual_size(n_tiles_n, n_tile_id, tile_size_n, short_tile_size_n);

            int stream_id = omp_get_thread_num();

            // possibly invoke some dgemms on CPU
            // dgemm_(&N, &N, &m_cpu, &n_cpu, &k_cpu, &one, a, &lda, b + ldb * n_gpu, &ldb, &beta, c + ldc * n_gpu, &ldc);

            // We let each thread perform all k-tiles for fixed m_tile_id and n_tile_id
            // so that the partial results can be reused and immediately reduced while on GPU.
            // This way, we only have to copy matrix C at most 2 times: at the beginning 
            // and at the very end, when all the k-tiles are computed.

            // loop over k-tiles
            for (int k_tile_id = 0; k_tile_id < n_tiles_k; k_tile_id++) {
                int actual_size_k = actual_size(n_tiles_k, k_tile_id, tile_size_k, short_tile_size_k);

                double new_beta = k_tile_id == 0 ? beta : 1.0;

            /*  Without the critical region, the following might happen:
             *  (since copy in the same direction is serialized even 
             *  if on different streams, as well as cuBLAS kernels).
             *
             *  stream 0: copy A,         copy B, Kernel A*B, copy C
             *  stream 1:         copy A,         copy B,     Kernel A*B, copy C
             *
             *  but the following is more efficient:
             *
             *  stream 0: copy A, copy B, Kernel A*B, copy C
             *  stream 1:                 copy A, B,  Kernel A*B, copy C <----->
             *                                                          difference
             *
             * For this reason, we want to enforce that coping A and B is never interrupted.
             * We can achieve this by letting the stream wait for the copy-event from another stream
             * (which induces the inter-stream synchronization and slows down the performance)
             * or we can block the host from launching copy kernels until the previous 
             * kernel has been finished. For this, we use a critical region and 
             * a shared copy-event object, on which the stream blocks.
             */
#pragma omp critical
                {
                    // streams[stream_id].wait_on_event(copy_event);
                    // make the host thread wait on the previous copy event
                    // (from other host thread which has the lock).
                    // this will not block the device but the host thread
                    copy_event.wait();
                    // copy next tile to device
                    copy_tile_to_device_async(a, a_device,
                            tile_size_m, tile_size_k,
                            short_tile_size_m, short_tile_size_k,
                            m, k,
                            n_tiles_m, n_tiles_k,
                            m_tile_id, k_tile_id,
                            stream_id, streams[stream_id]);

                    // copy next tile to device
                    copy_tile_to_device_async(b, b_device,
                            tile_size_k, tile_size_n,
                            short_tile_size_k, short_tile_size_n,
                            k, n,
                            n_tiles_k, n_tiles_n,
                            k_tile_id, n_tile_id,
                            stream_id, streams[stream_id]);

                    // copy next tile to device
                    if (k_tile_id == 0 && beta > 0) {
                        copy_tile_to_device_async(c, c_device,
                                tile_size_m, tile_size_n,
                                short_tile_size_m, short_tile_size_n,
                                m, n,
                                n_tiles_m, n_tiles_n,
                                m_tile_id, n_tile_id,
                                stream_id, streams[stream_id]);
                    }
                    copy_event = streams[stream_id].enqueue_event();
                }

                // perform dgemm
                cublasDgemm(get_cublas_handle(stream_id), CUBLAS_OP_N, CUBLAS_OP_N,
                        actual_size_m, actual_size_n, actual_size_k, &alpha,
                        &a_device[stream_id*offset_a], actual_size_m,
                        &b_device[stream_id*offset_b], actual_size_k, &new_beta,
                        &c_device[stream_id*offset_c], actual_size_m);

                // copy result back to host
                copy_tile_to_host_async(c_device, c,
                        tile_size_m, tile_size_n,
                        short_tile_size_m, short_tile_size_n,
                        m, n,
                        n_tiles_m, n_tiles_n,
                        m_tile_id, n_tile_id,
                        stream_id,
                        streams[stream_id]);

                // if this is the last event on this stream => record it
                if (m_tile_id == n_tiles_m - 1 && n_tile_id == n_tiles_n - 1)
                    last_event_on_stream[stream_id] = streams[stream_id].enqueue_event();
            }
        }
    }

    // block until all streams have finished all the work
    for (int stream_id = 0; stream_id < n_streams; ++stream_id) {
        int tid = omp_get_thread_num();
        last_event_on_stream[tid].wait();
    }

#ifdef DEBUG
    std::cout << "Total memory used on GPU = " << gpu_allocated_memory() << std::endl;
    std::cout << "Testing the result of GPU multiplication..." << std::endl;
    bool wrong = false;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double result_c = c[j * m + i];
            double real_result = 0;

            for (int x = 0; x < k; ++x) {
                real_result += a[x * m + i] * b[j * k + x];
            }

            wrong = wrong || (std::abs(result_c - real_result) > 1e-5);
            if (wrong) {
                std::cout << "WRONG RESULT: GPU c[" << i << ", " << j << "] = " << result_c << ", instead of " << real_result << std::endl;
            }
        }
    }

    if (!wrong) {
        std::cout << "Result correct on this rank" << std::endl;
    }
#endif
}
