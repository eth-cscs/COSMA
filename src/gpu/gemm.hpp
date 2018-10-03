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

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          double* a_intermediate, double* b_intermediate, double* c_intermediate,
          int m, int n, int k,
          int tile_size_m, int tile_size_n, int tile_size_k,
          double alpha, double beta);
