## Building COSMA

To build COSMA, do the following steps:
```bash
# clone the repository
git clone --recursive https://github.com/eth-cscs/COSMA.git
cd COSMA

# create a build directory within COSMA
mkdir build
cd build

# Choose which BLAS and SCALAPACK backends to use (e.g. MKL)
cmake -DCOSMA_BLAS=MKL -DCOSMA_SCALAPACK=MKL ..

# compile
make -j 8
```
> !! Note the *--recursive* flag !! 

Other important options that can be passed to `cmake` are the following:
- `COSMA_BLAS:` `MKL` (default), `OPENBLAS`, `CRAY_LIBSCI`, `CUSTOM`, `CUDA` or `ROCM`. Determines which backend will be used for the local matrix multiplication calls.
- `COSMA_SCALAPACK:` OFF (default), `MKL`, `CRAY_LIBSCI`, `CUSTOM`. If specified, `COSMA` will also provide ScaLAPACK wrappers, thus offering `pdgemm`, `psgemm`, `pzgemm` and `pcgemm` functions, which completely match the ScaLAPACK API.

## Building COSMA on Cray Systems

There are already prepared scripts for loading the necessary dependencies for COSMA on Cray-Systems:
- `Cray XC40` (CPU-only version): `source ./scripts/piz_daint_cpu.sh` loads `MKL` and other neccessary modules.
- `Cray XC50` (Hybrid version): `source ./scripts/piz_daint_gpu.sh` loads `cublas` and other necessary modules.

After the right modules are loaded, the instructions from the beginning of this file can be followed.

## Installing COSMA

To install do `make install`. 

> !! Note: To set custom installation directory use `CMAKE_INSTALL_PREFIX` when building. 

COSMA is CMake friendly and provides a cosmaConfig.cmake module for easy
integration into 3rd-party CMake projects with

```
find_package(cosma REQUIRED)
target_link_libraries( ... cosma::cosma)
```

COSMA's dependencies are taken care of internally, nothing else needs to be
linked. Make sure to set `CMAKE_INSTALL_PREFIX` to COSMA's installation directory
when building.

There is a rudimentary pkgconfig support; dependencies are handles explicitly by
consumers.
