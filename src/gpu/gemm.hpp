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

#define nstreams 3

void gpu_dgemm_(double* a, double* b, double* c,
          double* a_device, double* b_device, double* c_device,
          int m, int n, int k,
          double alpha, double beta);
