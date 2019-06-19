#!/bin/bash

module load daint-mc
module load CMake
module unload cray-libsci
module load intel # defines $MKLROOT

#echo $MKLROOT
#cc --version

# MKL_NUM_THREADS

# enable the dynamic linking and
# the asynchronous thread progressing
# MPICH (on Cray systems)
export CRAYPE_LINK_TYPE=dynamic
#export MPICH_NEMESIS_ASYNC_PROGRESS=MC
#export MPICH_MAX_THREAD_SAFETY=multiple
#export MPICH_GNI_ASYNC_PROGRESS_TIMEOUT=0

# setup the right compilers
export CC=`which cc`
export CXX=`which CC`

#rm -rf CMakeCache.txt CMakeFiles

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLA_VENDOR="Intel10_64lp" 

make

# Run tests
#
ctest
