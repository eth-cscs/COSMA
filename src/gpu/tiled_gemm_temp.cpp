#include "gemm.hpp"

void copy_tile(double* from, double* to,
        int tile_m, int tile_n,
        int tile_size,
        int actual_tile_m, int actual_tile_n,
        int m, int n,
        int p_row_tile, int p_col_tile,
        int col,
        int ibuff,
        bool global_to_pinned) {

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

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          double alpha, double beta) {

    // define parameters
    int tile_size_m = 3;
    int tile_size_n = 3;
    int tile_size_k = 3;

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
            if (iktile == n_tiles_k - 1 && short_tile_size_k > 0)
                actual_size_k = short_tile_size_k;
            else
                actual_size_k = tile_size_k;

            // loop over row tiles
            for (int irowtile = 0; irowtile < n_tiles_m; irowtile++) {
                if (irowtile == n_tiles_m - 1 && short_tile_size_m > 0)
                    actual_size_m = short_tile_size_m;
                else
                    actual_size_m = tile_size_m;

                // loop over column tiles
                for (int icoltile = 0; icoltile < n_tiles_n; icoltile++) {
                    if (icoltile == n_tiles_n - 1 && short_tile_size_n > 0)
                        actual_size_n = short_tile_size_n;
                    else
                        actual_size_n = tile_size_n;

                    if (itile >= nstreams) {
                        // block the host until this streams buffers are available
                        // (that is, all previous operations in this stream have completed)
                        bufferfilled[ibuff].wait();

                        // copy result in pinned buffer back to global matrix
//# pragma omp parallel for
                        for ( int j=0; j<actual_size_n; j++ ) {
                            copy_tile(pc, c, 
                                    tile_size_m, tile_size_n, 
                                    offset_c, 
                                    actual_size_m, actual_size_n, 
                                    m, n, 
                                    p_row_tile[ibuff], p_col_tile[ibuff], 
                                    j, ibuff, false);
                        }
                    }

                    // copy next tile to pinned buffer
//# pragma omp parallel for
                    for ( int j=0; j<actual_size_k; j++ ) {
                        copy_tile(a, pa,
                                tile_size_m, tile_size_k,
                                offset_a,
                                actual_size_m, actual_size_k,
                                m, k,
                                irowtile, iktile,
                                j, ibuff, true);
                    }

                    // copy tile data to device
//# pragma omp parallel for
                    for ( int j=0; j<actual_size_n; j++ ) {
                        copy_tile(b, pb,
                                tile_size_k, tile_size_n,
                                offset_b,
                                actual_size_k, actual_size_n,
                                k, n,
                                iktile, icoltile,
                                j, ibuff, true);
                    }

                    // copy next tile to pinned buffer
//# pragma omp parallel for
                    for ( int j=0; j<actual_size_n; j++ ) {
                        copy_tile(c, pc,
                                tile_size_m, tile_size_n,
                                offset_c,
                                actual_size_m, actual_size_n,
                                m, n,
                                irowtile, icoltile,
                                j, ibuff, true);
                    }

                    copy_to_device_async(&pa[ibuff*offset_a], &d_a[ibuff*offset_a],
                            actual_size_m*actual_size_k, myStreams[ibuff].stream());
                    copy_to_device_async(&pb[ibuff*offset_b], &d_b[ibuff*offset_b],
                            actual_size_k*actual_size_n, myStreams[ibuff].stream());
                    copy_to_device_async(&pc[ibuff*offset_c], &d_c[ibuff*offset_c],
                            actual_size_m*actual_size_n, myStreams[ibuff].stream());

                    // tell cuBLAS which stream to use
                    cublasSetStream(get_cublas_handle(), myStreams[ibuff].stream());

                    // perform dgemm
                    cublasDgemm (get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                            actual_size_m, actual_size_n, actual_size_k, &alpha,
                            &d_a[ibuff*offset_a], actual_size_m,
                            &d_b[ibuff*offset_b], actual_size_k, &beta,
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
//#  pragma omp parallel for
            for ( int j=0; j<actual_size_n; j++ ) {
                copy_tile(pc, c,
                        tile_size_m, tile_size_n,
                        offset_c,
                        actual_size_m, actual_size_n,
                        m, n,
                        p_row_tile[itile], p_col_tile[itile],
                        j, itile, false);
            }
        }
    }

    // cudaEvent_t t_end;
    // cudaEventRecord (t_end,0);
    // cudaEventSynchronize(t_end);
    // cudaEventDestroy(t_end);

    cudaFreeHost(pa);
    cudaFreeHost(pb);
    cudaFreeHost(pc);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
