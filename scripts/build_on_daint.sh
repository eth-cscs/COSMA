#!/bin/bash

module load daint-mc
module load CMake
module unload cray-libsci
module load intel # defines $MKLROOT

# Setup compilers
#
export CC=`which cc`
export CXX=`which CC`

# Enable dynamic linking
#
export CRAYPE_LINK_TYPE=dynamic

# Enable asynchronous thread progressing with MPICH (on Cray systems)
#
#export MPICH_NEMESIS_ASYNC_PROGRESS=MC
#export MPICH_MAX_THREAD_SAFETY=multiple
#export MPICH_GNI_ASYNC_PROGRESS_TIMEOUT=0

# Set number of threads
#
# export MKL_NUM_THREADS = ???

# Move to `build` directory if not there already 
# 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
COSMA_DIR="$( dirname $SCRIPT_DIR )" 
mkdir -p ${COSMA_DIR}/build 
cd ${COSMA_DIR}/build

# Clean CMake files
#
#rm -rf CMakeCache.txt CMakeFiles

# Configure COSMA
#
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_THREADING="GOMP"

  # DCMAKE_PREFIX_PATH= ???
  #-DBLA_VENDOR="Intel10_64lp" 

# Build all targets.
#
make all
