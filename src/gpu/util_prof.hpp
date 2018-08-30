#pragma once
#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <mutex>
///////////////////////////////////////////////////////////////////////////////
// nvprof profiler interface
///////////////////////////////////////////////////////////////////////////////

// global variables for managing access to profiler
bool is_running_nvprof = false;
std::mutex gpu_profiler_mutex;

inline
void start_nvprof() {
    std::lock_guard<std::mutex> guard(gpu_profiler_mutex);
    if (!is_running_nvprof) {
        cudaProfilerStart();
    }
    is_running_nvprof = true;
}

inline
void stop_nvprof() {
    std::lock_guard<std::mutex> guard(gpu_profiler_mutex);
    if (is_running_nvprof) {
        cudaProfilerStop();
    }
    is_running_nvprof = false;
}

