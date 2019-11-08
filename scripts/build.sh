#!/bin/bash

# !!! --------------------------------- !!!
# !!! Adjust the script to your system. !!!
# !!! --------------------------------- !!!

# Clean the build directory.
#
#rm -rf CMakeCache.txt CMakeFiles

# If MKL is used
#
#export MKLROOT=<FIXME>

# If GPU back end is used (Tiled-MM), set the following path:
#
#export CUDA_PATH=<FIXME> 

# Options
# ^^^^^^^
# 
# `CMAKE_BUILD_TYPE` := Debug|Release|Profile (default: Release)
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
# `COSMA_WITH_PROFILING`: = ON|OFF (default: OFF)
#    Enables profiling of COSMA with `semiprof`.
#
# BLAS (select one of:)
#
# `COSMA_BLAS` := MKL|OPENBLAS|NETLIB|CUDA|ROCM (default: MKL)
#
# `COSMA_SCALAPACK` := OFF|MKL|NETLIB (default: OFF)
#
#    Note: Mixing OpenMP runtimes results in performance issues. If you use 
#          COSMA within a large application, make sure that a single OpenMP
#          back end is used. If using GCC, that should be GNU OpenMP, except
#          on Mac. COSMA automically selects the right OpenMP runtime back end 
#          based on platform and compiler.
#
cmake <FIXME:cosma_source_dir> \
  -D CMAKE_INSTALL_PREFIX=<FIXME:cosma_install_dir> \

