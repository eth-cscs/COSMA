#!/bin/bash

###############################################################################
# Note: Adjust the script to your system or use it as a template for a new 
#       script.
###############################################################################

# Clean the build directory.
#
rm -rf CMakeCache.txt CMakeFiles

# Set MKL's root directory.
#
# See available options for MKL versions in `cmake/FindMKL.cmake`
#
export MKLROOT="TODO"

# Set the root directories of remaining dependencies.
#
GRID2GRID_ROOT="TODO"
OPTIONS_ROOT="TODO"
SEMIPROF_ROOT="TODO"

# Set the desired installation directory.
#
INSTALL_ROOT="TODO"

# Set the path to the COSMA source directory.
#
SRC_ROOT="TODO"

# Options:
# 
# `CMAKE_BUILD_TYPE` := Debug|Release
#
# `COSMA_WITH_PROFILING`: = ON|OFF (default:OFF)
#    Enables profiling of COSMA with `semiprof`.
#
# `COSMA_IS_OPENMPI` := ON|OFF (default:OFF)
# 
# `MKL_THREADING` := IOMP|GOMP|serial (default:serial)
#    IOMP is the Intel OpenMP back-end, GOMP is the GNU OpenMP back-end. When 
#    compiling with gcc, use GOMP. Mixing OpenMP runtimes results in performance 
#    issues.
#
# `MKL_USE_64BIT_INTEGERS := True|False (default:false)
#    Select the 32 bit or 64 bit MKL integer interface.
#    
# `MKL_MPI_TYPE` := OMPI|MPICH (default:no ScaLAPACK)
#    Enables MKL ScaLAPACK and selects the MPI backend to use. OMPI stands for 
#    OpenMPI. MPICH is also used for derivative implementations: Intel MPI, 
#    Cray MPI, etc.
#
cmake ${SRC_ROOT} \
  -D CMAKE_PREFIX_PATH="${GRID2GRID_ROOT};${OPTIONS_ROOT};${SEMIPROF_ROOT};" \
  -D CMAKE_INSTALL_PREFIX="${INSTALL_ROOT}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D MKL_THREADING="GOMP" \
  -D COSMA_WITH_PROFILING=ON \
  -D COSMA_IS_OPENMPI=ON

