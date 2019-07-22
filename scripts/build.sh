#!/bin/bash

# !!! NOTE: Adjust the script to your system. !!!

# Clean the build directory.
#
rm -rf CMakeCache.txt CMakeFiles

export MKLROOT=<TODO:mkl_root_dir>

# Options:
# 
# `CMAKE_BUILD_TYPE` := Debug|Release
#
# `COSMA_WITH_PROFILING`: = ON|OFF (default: OFF)
#    Enables profiling of COSMA with `semiprof`.
#
# `COSMA_WITH_TESTS`: = ON|OFF (default: ON if COSMA is not a subproject)
#    Enables tests.
#
# `COSMA_WITH_APPS`: = ON|OFF (default: ON if COSMA is not a subproject)
#    Enables miniapps.
#
# `COSMA_WITH_BENCHMARKS`: = ON|OFF (default: ON if COSMA is not a subproject)
#    Enables benchmarks.
#
# `COSMA_WITH_OPENMPI` := ON|OFF (default:OFF)
#    Only relevant for unit tests. Makes sure correct flags are pasts to tests.
# 
# `MKL_PARALLEL` := ON|OFF (default: ON)
#    IOMP is the Intel OpenMP back-end, GOMP is the GNU OpenMP back-end. When 
#    compiling with gcc, use GOMP. Mixing OpenMP runtimes results in performance 
#    issues.
#
# `MKL_64BIT := ON|OFF (default: OFF)
#    `ON` selects the 64 bit MKL integer interface.
#    
# `MKL_MPI_TYPE` := OMPI|MPICH (default: MPICH)
#    Enables MKL ScaLAPACK and selects the MPI backend to use. OMPI stands for 
#    OpenMPI. MPICH is also used for derivative implementations: Intel MPI, 
#    Cray MPI, etc. 
#    ON Apple only MPICH us supported.
#
cmake <TODO:cosma_source_dir> \
  -D CMAKE_INSTALL_PREFIX=<TODO:cosma_install_dir>\
  -D CMAKE_BUILD_TYPE="Release" \
  -D MKL_THREADING="GOMP" \
  -D COSMA_WITH_BENCHMARKS=OFF \
  -D COSMA_WITH_OPENMPI=ON

