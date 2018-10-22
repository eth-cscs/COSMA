#pragma once
#include <iostream>
#include <cmath>
#include <cstdio>
#include "util.hpp"
#include "cuda_stream.hpp"
#include "cuda_event.hpp"
#include "../blas.h"
#include <vector>
#include <omp.h>
#include <cstring>
#include "tile_description.hpp"
#include "cublas_handle.hpp"

/*
  **************************************
         TILE-COPYING METHODS
  **************************************

  Matrix in host memory is given in column-major form.
  Matrix from host memory is split into tiles and each
  tile is copied to device memory.

  A tile is a small block of the host memory,
  which is also in column major order.

  **************************************
   Example:
  **************************************
  Matrix:
  1 2 5 4
  4 2 3 5
  1 6 7 4
  9 0 1 2

  Host:
  pointer to -> 1 4 1 9 2 2 6 0 5 3 7 1 4 5 4 2

  Device (assuming tile dimensions are 2x2):
  Tile 00: pointer to -> 1 4 2 2
  Tile 10: pointer to -> 1 9 6 0
  Tile 01: pointer to -> 5 3 4 5
  Tile 11: pointer to -> 7 1 4 2
       ^
  Tile id = <row_id><col_id>

  **************************************
      MEMORY WITH MULTIPLE-STREAMS
  **************************************
  On the device, N_STREAMS*tile_size memory is preallocated,
  such that each stream has a separate piece of memory.

  device memory (assuming 3 streams)
            -----------> stream offset
            ___________________________________
  array:   | TILE SIZE | TILE SIZE | TILE SIZE |
            ___________________________________
              stream 1    stream 2    stream 3
          ^
    device pointer

  Observe that the device pointer points to the beginning
  of the pre-allocated device buffer. Therefore, each stream
  has to add offset stream_id * TILE_SIZE to get the pointer
  to device tile it is in charge of.

  copy_tile methods copy a single tile between host<->device.

  **************************************
               ARGUMENTS
  **************************************
  - from: pointer to the beginning of the source memory 
  - to: pointer to the beginning of the destination memory.
  - tile_m: number of rows of a single tile
  - tile_n: number of columns of a single tile
  - short_tile_m: if the tile_m does not perfectly divide the global
                  matrix dimension m, then this is the remainder,
                  i.e. the dimension of the last tile that is
                  degenerated (different than others).
  - short_tile_n: if the tile_n does not perfectly divide the global
                  matrix dimension n, then this is the remainder,
                  i.e. the dimension of the last tile that is
                  degenerated (different than others).
  - m: number of rows of the host matrix
  - n: number of columns of the host matrix
  - n_tiles_m: number of tiles in dimension m
  - n_tiles_n: number of tiles in dimension n
  - p_row_tile: row id of a tile that should be copied
  - p_col_tile: row id of a tile that should be copied
  - stream_id: stream id (0-based)
  - stream: stream on which to schedule the copy task
*/

void copy_tile_to_device_async(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id,
        cuda_stream& stream);

void copy_tile_to_host_async(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id,
        cuda_stream& stream);

void copy_tile_to_device(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id);

void copy_tile_to_host(double* from, double* to,
        int tile_m, int tile_n,
        int short_tile_m, int short_tile_n,
        int m, int n,
        int n_tiles_m, int n_tiles_n,
        int p_row_tile, int p_col_tile,
        int stream_id);

// **************************************
//        TILED-GEMM ON GPU
// **************************************
void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          int tile_size_m, int tile_size_n, int tile_size_k,
          double alpha, double beta);

