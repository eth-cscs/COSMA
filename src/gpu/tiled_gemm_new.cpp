#include "gemm.hpp"

int actual_size(int n_tiles, int tile_id, int tile_length, int tile_remainder) {
    bool last_tile = tile_id == n_tiles - 1;
    bool not_divisible = tile_remainder > 0;

    return last_tile && not_divisible ? tile_remainder : tile_length;
}

void copy_tile(double* from, double* to,
        int tile_m, int tile_n,
        int tile_size,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int ibuff,
        bool global_to_pinned) {

    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

# pragma omp parallel for num_threads(N_STREAMS)
    for (int col = 0; col < actual_tile_n; ++col) {
        // pinned buffers offsets
        int tile_offset = ibuff*tile_size;
        int el_offset = col * actual_tile_m;
        int offset = tile_offset + el_offset;

        int tile_offset_global = (p_col_tile*tile_n + col) * m;
        int el_offset_global = p_row_tile * tile_m;
        int offset_global = tile_offset_global + el_offset_global;

        int offset_from = global_to_pinned ? offset_global : offset;
        int offset_to = global_to_pinned ? offset : offset_global;

        std::memcpy(to + offset_to, from + offset_from, actual_tile_m * sizeof(double));
    }
}

void copy_tile_to_device_async(double* from, double* to,
        int tile_m, int tile_n,
        int tile_size,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int ibuff,
        cuda_stream& stream) {

    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    // pinned buffers offsets
    int offset_to = ibuff * tile_size;

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
        int tile_size,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int ibuff) {

    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    // pinned buffers offsets
    int offset_to = ibuff * tile_size;

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
        int tile_size,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int ibuff,
        cuda_stream& stream) {

    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    int offset_from = ibuff * tile_size;

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
        int tile_size,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int ibuff) {

    int actual_tile_m = actual_size(n_tiles_m, p_row_tile, tile_m, short_tile_m);
    int actual_tile_n = actual_size(n_tiles_n, p_col_tile, tile_n, short_tile_n);

    int offset_from = ibuff * tile_size;

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
        double* a_intermediate, double* b_intermediate, double* c_intermediate,
        int m, int n, int k,
        int tile_size_m, int tile_size_n, int tile_size_k,
        double alpha, double beta) {
    // omp_set_nested(1);
    //
    // cudaHostRegister(a, m * k * sizeof(double), cudaHostRegisterDefault);
    // cudaHostRegister(b, k * n * sizeof(double), cudaHostRegisterDefault);
    // cudaHostRegister(c, m * n * sizeof(double), cudaHostRegisterDefault);

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

    int n_streams = std::min(N_STREAMS, n_tiles_m * n_tiles_n * n_tiles_k);

    std::vector<cuda_stream> myStreams(n_streams);

    std::vector<cuda_event> bufferfilled(n_streams);

    // caches for indices of previous tiles in streams
    std::vector<int> p_row_tile(n_streams);
    std::vector<int> p_col_tile(n_streams);

    cuda_event copy_event;

    // PERFORM MULTIPLICATION
    // int ibuff = 0;
    int itile_mn = 0;

#pragma omp parallel for collapse(2) num_threads(n_streams)
    // loop over row tiles
    for (int irowtile = 0; irowtile < n_tiles_m; irowtile++) {
        // loop over column tiles
        for (int icoltile = 0; icoltile < n_tiles_n; icoltile++) {
            int actual_size_m = actual_size(n_tiles_m, irowtile, tile_size_m, short_tile_size_m);
            int actual_size_n = actual_size(n_tiles_n, icoltile, tile_size_n, short_tile_size_n);

            int ibuff = omp_get_thread_num();
            // bool first_event = true;

            // if (!first_event) {
            //     bufferfilled[ibuff].wait();
            //     // copy result in pinned buffer back to global matrix
            //     copy_tile(c_intermediate, c, 
            //             tile_size_m, tile_size_n, 
            //             offset_c, 
            //             short_tile_size_m, short_tile_size_n,
            //             m, n,
            //             n_tiles_m, n_tiles_n,
            //             p_row_tile[ibuff], p_col_tile[ibuff],
            //             ibuff, false);
            // }

            //dgemm_(&N, &N, &m_cpu, &n_cpu, &k_cpu, &one, a, &lda, b + ldb * n_gpu, &ldb, &beta, c + ldc * n_gpu, &ldc);

            // cuda_event copy_event;

            // loop over inner tile dimension
            for (int iktile = 0; iktile < n_tiles_k; iktile++) {
                int actual_size_k = actual_size(n_tiles_k, iktile, tile_size_k, short_tile_size_k);

                double new_beta = iktile == 0 ? beta : 1.0;

#pragma omp critical
                {
                myStreams[ibuff].wait_on_event(copy_event);
                // copy_event.wait();
                // copy next tile to device
                copy_tile_to_device_async(a, a_device,
                        tile_size_m, tile_size_k,
                        offset_a,
                        short_tile_size_m, short_tile_size_k,
                        m, k,
                        n_tiles_m, n_tiles_k,
                        irowtile, iktile,
                        ibuff, myStreams[ibuff]);

                // copy next tile to device
                copy_tile_to_device_async(b, b_device,
                        tile_size_k, tile_size_n,
                        offset_b,
                        short_tile_size_k, short_tile_size_n,
                        k, n,
                        n_tiles_k, n_tiles_n,
                        iktile, icoltile,
                        ibuff, myStreams[ibuff]);

                // copy next tile to device
                if (iktile == 0 && beta > 0) {
                    copy_tile_to_device_async(c, c_device,
                            tile_size_m, tile_size_n,
                            offset_c,
                            short_tile_size_m, short_tile_size_n,
                            m, n,
                            n_tiles_m, n_tiles_n,
                            irowtile, icoltile,
                            ibuff, myStreams[ibuff]);
                }
                copy_event = myStreams[ibuff].enqueue_event();
                }

#pragma omp critical 
            {
                // tell cuBLAS which stream to use
                cublasSetStream(get_cublas_handle(), myStreams[ibuff].stream());

                // perform dgemm
                cublasDgemm(get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        actual_size_m, actual_size_n, actual_size_k, &alpha,
                        &a_device[ibuff*offset_a], actual_size_m,
                        &b_device[ibuff*offset_b], actual_size_k, &new_beta,
                        &c_device[ibuff*offset_c], actual_size_m);
            }
            }

            // copy result back to host
            copy_tile_to_host_async(c_device, c,
                    tile_size_m, tile_size_n,
                    offset_c,
                    short_tile_size_m, short_tile_size_n,
                    m, n,
                    n_tiles_m, n_tiles_n,
                    irowtile, icoltile,
                    ibuff,
                    myStreams[ibuff]);

            // this event will signal when the D2H copy of the result has completed
            if (irowtile == n_tiles_m - 1 && icoltile == n_tiles_n - 1)
                bufferfilled[ibuff] = myStreams[ibuff].enqueue_event();
            //first_event = false;

            p_row_tile[ibuff] = irowtile;
            p_col_tile[ibuff] = icoltile;
            // ibuff = (ibuff + 1) % n_streams;
        }
    }

// #pragma omp parallel for
    for (int ibuff = 0; ibuff < n_streams; ++ibuff) {
        bufferfilled[ibuff].wait();
        // copy result in pinned buffer back to global matrix
        // copy_tile(c_intermediate, c, 
        //         tile_size_m, tile_size_n, 
        //         offset_c, 
        //         short_tile_size_m, short_tile_size_n,
        //         m, n,
        //         n_tiles_m, n_tiles_n,
        //         p_row_tile[ibuff], p_col_tile[ibuff],
        //         ibuff, false);
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

    // cudaHostUnregister(a);
    // cudaHostUnregister(b);
    // cudaHostUnregister(c);
}
