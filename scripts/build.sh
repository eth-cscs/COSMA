#!/bin/bash

# !!! NOTE: Adjust the script to your system. !!!

# Clean the build directory.
#
rm -rf CMakeCache.txt CMakeFiles

export MKLROOT=<TODO:mkl_root_dir>

# If GPU back end is used (Tiled-MM), set the following path:
#
#export CUDA_PATH=<TODO> 

# Options:
# 
# `CMAKE_BUILD_TYPE` := Debug|Release (default: Release)
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
# `COSMA_WITH_TILEDMM` := ON|OFF (default: OFF)
#    If `ON` uses the TiledMM (submodule) GPU gemm back-end instead of MKL.
# 
#
#  Note: The followint MKL options are only relevant if the MKL back end 
#        is used.
#
# `MKL_PARALLEL` := ON|OFF (default: ON)
#    Uses the Intel/GNU OpenMP back end. If `OFF`, uses sequential MKL.
#
#    Note: Mixing OpenMP runtimes results in performance issues. If you use 
#          COSMA within a large application, make sure that a single OpenMP
#          back end is used. If using GCC, that should be GNU OpenMP, except
#          on Mac. COSMA automically selects the right OpenMP runtime back end 
#          based on platform and compiler.
#
# `MKL_64BIT` := ON|OFF (default: OFF)
#    `ON` selects the 64 bit MKL integer interface.
#    
# `MKL_MPI_TYPE` := OMPI|MPICH (default: MPICH)
#    Only relevant if ScaLAPACK was found. OMPI stands for OpenMPI. MPICH is 
#    also used for derivative implementations: Intel MPI, Cray MPI, etc. On Mac 
#    only MPICH is supported.
#
cmake <TODO:cosma_source_dir> \
  -D CMAKE_INSTALL_PREFIX=<TODO:cosma_install_dir> \

