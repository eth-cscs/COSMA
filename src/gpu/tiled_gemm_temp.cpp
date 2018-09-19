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

# pragma omp parallel for
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

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          double alpha, double beta) {

    // define parameters
    int tile_size_m = 3000;
    int tile_size_n = 3000;
    int tile_size_k = 3000;

    tile_size_m = std::min(tile_size_m, m);
    tile_size_n = std::min(tile_size_n, n);
    tile_size_k = std::min(tile_size_k, k);

    // short tile sizes (when dim not divisible by tile)
    int short_tile_size_m = m % tile_size_m;
    int short_tile_size_n = n % tile_size_n;
    int short_tile_size_k = k % tile_size_k;

    // create communcations arrays
    double *pa = malloc_pinned<double>(tile_size_m * tile_size_k * nstreams);
    double *pb = malloc_pinned<double>(tile_size_k * tile_size_n * nstreams);
    double *pc = malloc_pinned<double>(tile_size_m * tile_size_n * nstreams);
    //double *pa = malloc_device<double>(tile_size_m * tile_size_k * nstreams);
    //double *pb = malloc_device<double>(tile_size_k * tile_size_n * nstreams);
    //double *pc = malloc_device<double>(tile_size_m * tile_size_n * nstreams);

    // allocate space on device - 3 tiles for a, b, c
    double *d_a = malloc_device<double>(tile_size_m * tile_size_k * nstreams);
    double *d_b = malloc_device<double>(tile_size_k * tile_size_n * nstreams);
    double *d_c = malloc_device<double>(tile_size_m * tile_size_n * nstreams);

    int offset_a = tile_size_m * tile_size_k;
    int offset_b = tile_size_k * tile_size_n;
    int offset_c = tile_size_m * tile_size_n;

    int n_tiles_m = (int) std::ceil(1.0 * m / tile_size_m);
    int n_tiles_n = (int) std::ceil(1.0 * n / tile_size_n);
    int n_tiles_k = (int) std::ceil(1.0 * k / tile_size_k);

    std::vector<cuda_stream> myStreams(nstreams);

    std::vector<cuda_event> bufferfilled(nstreams);

    // caches for indices of previous tiles in streams
    std::vector<int> p_row_tile(nstreams);
    std::vector<int> p_col_tile(nstreams);


    // PERFORM MULTIPLICATION
    {
        int ibuff = 0;
        int itile = 0;
        int actual_size_k, actual_size_m, actual_size_n;

        // loop over inner tile dimension
        for (int iktile = 0; iktile < n_tiles_k; iktile++) {
            actual_size_k = actual_size(n_tiles_k, iktile, tile_size_k, short_tile_size_k);

            double new_beta = iktile == 0 ? beta : 1.0;

            // loop over row tiles
            for (int irowtile = 0; irowtile < n_tiles_m; irowtile++) {
                actual_size_m = actual_size(n_tiles_m, irowtile, tile_size_m, short_tile_size_m);

                // loop over column tiles
                for (int icoltile = 0; icoltile < n_tiles_n; icoltile++) {
                    actual_size_n = actual_size(n_tiles_n, icoltile, tile_size_n, short_tile_size_n);

                    if (itile >= nstreams) {
                        // block the host until this streams buffers are available
                        // (that is, all previous operations in this stream have completed)
                        bufferfilled[ibuff].wait();

                        // copy result in pinned buffer back to global matrix
                        copy_tile(pc, c, 
                                tile_size_m, tile_size_n, 
                                offset_c, 
                                short_tile_size_m, short_tile_size_n,
                                m, n,
                                n_tiles_m, n_tiles_n,
                                p_row_tile[ibuff], p_col_tile[ibuff], 
                                ibuff, false);
                    }

                    // copy next tile to pinned buffer
                    copy_tile(a, pa,
                            tile_size_m, tile_size_k,
                            offset_a,
                            short_tile_size_m, short_tile_size_k,
                            m, k,
                            n_tiles_m, n_tiles_k,
                            irowtile, iktile,
                            ibuff, true);

                    // copy tile data to device
                    copy_tile(b, pb,
                            tile_size_k, tile_size_n,
                            offset_b,
                            short_tile_size_k, short_tile_size_n,
                            k, n,
                            n_tiles_k, n_tiles_n,
                            iktile, icoltile,
                            ibuff, true);

                    if (new_beta > 0) {
                        // copy next tile to pinned buffer
                        copy_tile(c, pc,
                                tile_size_m, tile_size_n,
                                offset_c,
                                short_tile_size_m, short_tile_size_n,
                                m, n,
                                n_tiles_m, n_tiles_n,
                                irowtile, icoltile,
                                ibuff, true);
                    }

                    copy_to_device_async(&pa[ibuff*offset_a], &d_a[ibuff*offset_a],
                            actual_size_m*actual_size_k, myStreams[ibuff].stream());
                    copy_to_device_async(&pb[ibuff*offset_b], &d_b[ibuff*offset_b],
                            actual_size_k*actual_size_n, myStreams[ibuff].stream());
                    if (new_beta > 0) {
                        copy_to_device_async(&pc[ibuff*offset_c], &d_c[ibuff*offset_c],
                                actual_size_m*actual_size_n, myStreams[ibuff].stream());
                    }

                    // tell cuBLAS which stream to use
                    cublasSetStream(get_cublas_handle(), myStreams[ibuff].stream());

                    // perform dgemm
                    cublasDgemm (get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                            actual_size_m, actual_size_n, actual_size_k, &alpha,
                            &d_a[ibuff*offset_a], actual_size_m,
                            &d_b[ibuff*offset_b], actual_size_k, &new_beta,
                            &d_c[ibuff*offset_c], actual_size_m);

                    p_row_tile[ibuff] = irowtile;
                    p_col_tile[ibuff] = icoltile;

                    // copy result back to host
                    copy_to_host_async(&d_c[ibuff*offset_c], &pc[ibuff*offset_c],
                            actual_size_m*actual_size_n, myStreams[ibuff].stream());

                    // this event will signal when the D2H copy of the result has completed
                    bufferfilled[ibuff] = myStreams[ibuff].enqueue_event();

                    // update buffer / stream
                    ibuff++;
                    ibuff = ibuff%nstreams;
                    itile++;

                }
            }
        }

        for ( itile=0; itile < nstreams; itile ++ ) {
            // make sure that buffers are free
            cudaStreamSynchronize (myStreams[itile].stream());

            // copy result in pinned buffer back to source
            copy_tile(pc, c,
                    tile_size_m, tile_size_n,
                    offset_c,
                    short_tile_size_m, short_tile_size_n,
                    m, n,
                    n_tiles_m, n_tiles_n,
                    p_row_tile[itile], p_col_tile[itile],
                    itile, false);
        }
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

    cudaFreeHost(pa);
    cudaFreeHost(pb);
    cudaFreeHost(pc);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
