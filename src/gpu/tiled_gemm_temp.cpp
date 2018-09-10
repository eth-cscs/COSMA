#pragma once
#include <iostream>
#include <cmath>
#include <cstdio>
#include "util.hpp"
#include "cuda_stream.hpp"
#include "cuda_event.hpp"
#include "../blas.h"
#include <omp.h>

#define nstreams 4

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          double alpha, double beta) {

    // define parameters
    int tile_size_m = 4096;
    int tile_size_n = 4096;
    int tile_size_k = 4096;
    // short tile sizes (when dim not divisible by tile)
    int short_tile_size_m = m % reg_tile_size_m;
    int short_tile_size_n = n % reg_tile_size_n;
    int short_tile_size_k = k % reg_tile_size_k;

    // create communcations arrays
    double *pa = malloc_pinned<double>(tile_size_m * tile_size_k * nstreams);
    double *pb = malloc_pinned<double>(tile_size_k * tile_size_n * nstreams);
    double *pc = malloc_pinned<double>(tile_size_m * tile_size_n * nstreams);

    // create a handle to cuBlas
    cublasHandle_t cublasHandle = get_cublas_handle();

    // allocate space on device - 3 tiles for a, b, c
    double *d_a = malloc_device<double>(tile_size_m * tile_size_k * nstreams);
    double *d_b = malloc_device<double>(tile_size_k * tile_size_n * nstreams);
    double *d_c = malloc_device<double>(tile_size_m * tile_size_n * nstreams);

    int offset_a = tile_size_m * tile_size_k;
    int offset_b = tile_size_k * tile_size_n;
    int offset_c = tile_size_m * tile_size_n;

    int n_tiles_m = (int) std::ceil(m / tile_size_m);
    int n_tiles_n = (int) std::ceil(n / tile_size_n);
    int n_tiles_k = (int) std::ceil(k / tile_size_k);

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
                        cudaEventSynchronize ( bufferfilled[ibuff] );

                        // copy result in pinned buffer back to global matrix
# pragma omp parallel for
                        for ( int i=0; i<actual_size_m; i++ ) {
                            for ( int j=0; j<actual_size_n; j++ ) {
                                // pinned buffers offsets
                                int tile_offset_pinned = ibuff*offset_c;
                                int el_offset_pinned = j * tile_size_m + i;
                                // input matrix offsets
                                int tile_offset_global = (p_col_tile[ibuff] * tile_size_n + j) * m;
                                int el_offset_global = p_row_tile[ibuff] * tile_size_m + i;
                                // pindded buffer -> global matrix
                                // if statement ensures the correctness in cases when tile sizes don't 
                                // perfectly divide the dimension of that matrix
                                if (tile_offset_global + el_offset_global < m * n) {
                                    c[tile_offset_global + el_offset_global] = pc[tile_offset + col_major_offset];
                                }
                            }
                        }
                    }

                    // copy next tile to pinned buffer
# pragma omp parallel for
                    for ( int i=0; i<actual_size_m; i++ ) {
                        for ( int j=0; j<actual_size_k; j++ ) {
                            // pinned buffers offsets
                            int tile_offset_pinned = ibuff*offset_a;
                            int el_offset_pinned = j * tile_size_m + i;
                            // input matrices offsets
                            int tile_offset_global = (iktile * tile_size_k + j) * m;
                            int el_offset_global = irowtile * tile_size_m + i;
                            // global -> pinned
                            if (tile_offset_global + el_offset_global < m * k) {
                                pa[tile_offset_pinned + el_offset_pinned] = a[tile_offset_global + el_offset_global];
                            }
                        }
                    }
# pragma omp parallel for
                    for ( int i=0; i<actual_size_k; i++ ) {
                        for ( int j=0; j<actual_size_n; j++ ) {
                            // pinned buffers offsets
                            int tile_offset_pinned = ibuff*offset_b;
                            int el_offset_pinned = j * tile_size_k + i;
                            // input matrices offsets
                            int tile_offset_global = (icoltile * tile_size_n + j) * k;
                            int el_offset_global = iktile * tile_size_k + i;
                            // global -> pinned
                            if (tile_offset_global + el_offset_global < k * n) {
                                pb[tile_offset_pinned + el_offset_pinned] = b[tile_offset_global + el_offset_global];
                            }
                        }
                    }
# pragma omp parallel for
                    for ( int i=0; i<actual_size_m; i++ ) {
                        for ( int j=0; j<actual_size_n; j++ ) {
                                // pinned buffers offsets
                                int tile_offset_pinned = ibuff*offset_c;
                                int el_offset_pinned = j * tile_size_m + i;
                                // input matrix offsets
                                int tile_offset_global = (icoltile * tile_size_n + j) * m;
                                int el_offset_global = irowtile * tile_size_m + i;
                                // global -> pinned
                                if (tile_offset_global + el_offset_global < m * n) {
                                    c[tile_offset_global + el_offset_global] = pc[tile_offset + col_major_offset];
                                }
                        }
                    }

                    // copy tile data to device
                    copy_to_device_async(&pa[ibuff*offset_a], &d_a[ibuff*offset_a],
                            actual_size_m*actual_size_k, myStreams[ibuff].stream());
                    copy_to_device_async(&pb[ibuff*offset_b], &d_b[ibuff*offset_b],
                            actual_size_k*actual_size_n, myStreams[ibuff].stream());
                    copy_to_device_async(&pc[ibuff*offset_c], &d_c[ibuff*offset_c],
                            actual_size_m*actual_size_n, myStreams[ibuff].stream());

                    // tell cuBLAS which stream to use
                    cublasSetStream(cublasHandle, myStreams[ibuff].stream());

                    // perform dgemm
                    cublasDgemm (cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            actual_size_m, actual_size_n, actual_size_k, &alpha,
                            &d_a[ibuff*offset_a], actual_size_m,
                            &d_b[ibuff*offset_b], actual_size_k, &beta,
                            &d_c[ibuff*offset_c], actual_size_m);

                    prowtile[ibuff] = irowtile;
                    pcoltile[ibuff] = icoltile;

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
# pragma omp parallel for
            for ( int i=0; i<actual_size_m; i++ ) {
                for ( int j=0; j<actual_size_n; j++ ) {
                    // pinned buffers offsets
                    int tile_offset_pinned = ibuff*offset_c;
                    int el_offset_pinned = j * tile_size_m + i;
                    // input matrix offsets
                    int tile_offset_global = (p_col_tile[ibuff] * tile_size_n + j) * m;
                    int el_offset_global = p_row_tile[ibuff] * tile_size_m + i;
                    // pindded buffer -> global matrix
                    // if statement ensures the correctness in cases when tile sizes don't 
                    // perfectly divide the dimension of that matrix
                    if (tile_offset_global + el_offset_global < m * n) {
                        c[tile_offset_global + el_offset_global] = pc[tile_offset + col_major_offset];
                    }
                }
            }
        }
    }

    cudaFreeHost(pa);
    cudaFreeHost(pb);
    cudaFreeHost(pc);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
