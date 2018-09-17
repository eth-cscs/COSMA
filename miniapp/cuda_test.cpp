#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

int main(int argc, char** argv) {
    int idevice = 0;
    cudaSetDevice(idevice);
    cudaDeviceProp dprops;
    cudaGetDeviceProperties( &dprops, idevice );

    printf ("\nDevice name = %s, with compute capability %d.%d \n",
            dprops.name, dprops.major, dprops.minor);

    cublasHandle_t cublas_handle;

    cublasCreate(&cublas_handle);

    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(status) << std::endl;
    }

    cublasDestroy(cublas_handle);
}
