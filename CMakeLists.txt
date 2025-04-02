cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(cosma
  DESCRIPTION "Communication Optimal Matrix Multiplication"
  HOMEPAGE_URL "https://github.com/eth-cscs/COSMA"
  VERSION 2.6.6
  LANGUAGES CXX)

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(cmake/build_type.cmake)
include(cmake/adjust_mpiexec_flags.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES") # always write compile_commands.json

# Options
#

set(COSMA_GPU_BACKENDS_LIST "CUDA" "ROCM")
set(COSMA_SCALAPACK_LIST "OFF" "MKL" "CRAY_LIBSCI" "NVPL" "CUSTOM")
set(COSMA_BLAS_LIST   "auto" "MKL" "OPENBLAS" "CRAY_LIBSCI" "NVPL" "CUSTOM" "BLIS" "ATLAS" "CUDA" "ROCM" "OFF")
option(COSMA_WITH_TESTS "Generate the test target." ON)
option(COSMA_WITH_APPS "Generate the miniapp targets." ON)
option(COSMA_WITH_BENCHMARKS "Generate the benchmark targets." ON)
option(COSMA_WITH_PROFILING "Enable profiling." OFF)
option(COSMA_WITH_NCCL "Use NCCL as communication backend." OFF)
option(COSMA_WITH_RCCL "Use RCCL as communication backend." OFF)
option(COSMA_WITH_GPU_AWARE_MPI "Use gpu-aware MPI for communication." OFF)
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)
set(COSMA_SCALAPACK "OFF" CACHE STRING "scalapack implementation. Can be MKL, CRAY_LIBSCI, NVPL, CUSTOM or OFF.")
set(COSMA_BLAS "OFF" CACHE STRING "Blas library for computations on host or GPU")

set(COSMA_BLAS_VENDOR "OFF")
set(COSMA_GPU_BACKEND "OFF")

set_property(CACHE COSMA_SCALAPACK PROPERTY STRINGS ${COSMA_SCALAPACK_LIST})
set_property(CACHE COSMA_BLAS PROPERTY STRINGS ${COSMA_BLAS_LIST})

# we keep the old cosma behavior of indicating GPU support as a blas
# implementation. We have to sort out what we should find for the FindBLAS and
# GPU supports since they are treated as separate components

if(COSMA_BLAS STREQUAL "OFF")
    message(FATAL_ERROR "A Blas implementation is needed when running on CPU only: choices are : auto, MKL, OPENBLAS, CRAY_LIBSCI, NVPL, CUSTOM, BLIS, ATLAS, FLEXIBLAS, ARMPL, GenericBLAS, CUDA or ROCM")
endif()

if (COSMA_BLAS MATCHES "CUDA|ROCM")
  set(COSMA_GPU_BACKEND ${COSMA_BLAS})
else()
  set(COSMA_BLAS_VENDOR ${COSMA_BLAS})
endif()

if ((COSMA_WITH_NCCL OR COSMA_WITH_RCCL) AND NOT COSMA_GPU_BACKEND IN_LIST COSMA_GPU_BACKENDS_LIST)
  message(FATAL_ERROR "NCCL (RCCL) can only be used with the GPU backend set to CUDA (ROCM).")
endif()

if (COSMA_WITH_GPU_AWARE_MPI AND NOT COSMA_GPU_BACKEND IN_LIST COSMA_GPU_BACKENDS_LIST)
  message(FATAL_ERROR "GPU-aware MPI can only be used with the GPU backend set to CUDA or ROCM.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Dependencies
# MPI
set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
find_package(MPI COMPONENTS CXX REQUIRED)
adjust_mpiexec_flags()

# check if scalapack backend is valid
message(STATUS "Selected SCALAPACK backend for COSMA: ${COSMA_SCALAPACK}")

if(NOT COSMA_SCALAPACK IN_LIST COSMA_SCALAPACK_LIST)
  message(FATAL_ERROR "Invalid value for COSMA_SCALAPACK!")
endif()

# the blas targets are only defined when COSMA_SCALAPACK is ON whatever value of COSMA_GPU_BACKEND
if (NOT COSMA_SCALAPACK MATCHES "OFF")
  if (COSMA_SCALAPACK MATCHES "MKL" OR COSMA_SCALAPACK MATCHES "CRAY_LIBSCI" OR COSMA_SCALAPACK MATCHES "NVPL")
    set(COSMA_BLAS_VENDOR ${COSMA_SCALAPACK})
  else()
    set(COSMA_BLAS_VENDOR "auto")
  endif()
endif()


if (NOT COSMA_BLAS_VENDOR MATCHES "OFF|CUDA|ROCM")
  find_package(Blas REQUIRED)
endif()

if (NOT COSMA_SCALAPACK MATCHES "OFF")
  find_package(SCALAPACK REQUIRED)
endif ()

set(COSTA_WITH_PROFILING ${COSMA_WITH_PROFILING} CACHE INTERNAL "")
set(COSTA_SCALAPACK ${COSMA_SCALAPACK} CACHE INTERNAL "")

FetchContent_Declare(
  costa
  GIT_REPOSITORY https://github.com/eth-cscs/costa.git
  GIT_TAG        03847e66f05ad4a1eb371b85be628e218ce46f11 # v2.2.3
  FIND_PACKAGE_ARGS NAMES costa
)
# the joy of fetch_content. if we build costa and cosma together
# fetch_content will pick up the FindSCALAPACK from cosma NOT costa.
if (NOT TARGET costa::scalapack::scalapack AND NOT COSMA_SCALAPACK MATCHES "OFF")
  add_library(costa::scalapack::scalapack ALIAS cosma::scalapack::scalapack)
endif ()

FetchContent_MakeAvailable(costa)

# these are only GPU-backends
if (COSMA_GPU_BACKEND MATCHES "CUDA|ROCM")
  set(TILEDMM_GPU_BACKEND ${COSMA_GPU_BACKEND} CACHE INTERNAL "")
  FetchContent_Declare(
    Tiled-MM
    GIT_REPOSITORY https://github.com/eth-cscs/Tiled-MM.git
    GIT_TAG      0eb75179e670a04c649b50ae5e91bb71b43e4d06 # v2.3.2
    FIND_PACKAGE_ARGS NAMES tiled-MM
  )
  FetchContent_MakeAvailable(Tiled-MM)

  if (COSMA_WITH_NCCL)
    find_package(CUDAToolkit REQUIRED)
    find_package(NCCL REQUIRED)
  elseif (COSMA_WITH_RCCL)
    find_package(hip REQUIRED)
    find_package(rccl REQUIRED)
  endif()

  if (NOT TARGET Tiled-MM::Tiled-MM)
      message("Tiled-mm target not found")
  endif ()

endif()

if (COSMA_WITH_PROFILING)
  FetchContent_Declare(
    semiprof
    GIT_REPOSITORY  https://github.com/bcumming/semiprof.git
    GIT_TAG         f132142ff2215dfa073e416fa7911d8877d62752
    FIND_PACKAGE_ARGS NAMES semiprof

  )
  FetchContent_MakeAvailable(semiprof)
endif ()

if (COSMA_WITH_TESTS OR COSMA_WITH_APPS)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        4bf61f08697b110d9e3991864650a405b3dd515d # v3.2.1
    FIND_PACKAGE_ARGS NAMES cxxopts
  )
  FetchContent_MakeAvailable(cxxopts)
endif()

if (NOT TARGET Tiled-MM::Tiled-MM)
      message("Tiled-mm target not found")
endif ()
# preserve rpaths when installing and make the install folder relocatable
# use `CMAKE_SKIP_INSTALL_RPATH` to skip this
# https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
# skip RPATH if COSMA is installed to system directories
if(isSystemDir STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  if(APPLE)
    set(basePoint @loader_path)
  else()
    set(basePoint $ORIGIN)
  endif()
  file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH ${basePoint} ${basePoint}/${relDir})
endif()

# COSMA
#
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_subdirectory(src/cosma)

install(DIRECTORY "${cosma_SOURCE_DIR}/src/cosma"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.hpp")

write_basic_package_version_file(
  "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
  VERSION ${cosma_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_file("${cosma_SOURCE_DIR}/cmake/cosma.pc.in"
  "${cosma_BINARY_DIR}/cosma.pc"
  @ONLY)

configure_file("${cosma_SOURCE_DIR}/cmake/cosmaConfig.cmake.in"
  "${cosma_BINARY_DIR}/cosmaConfig.cmake"
  @ONLY)

write_basic_package_version_file(
  "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
  VERSION "${cosma_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(FILES "${cosma_BINARY_DIR}/cosmaConfig.cmake"
  "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
  "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindMKL.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindNVPL.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindBlas.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindSCALAPACK.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindOPENBLAS.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindFLEXIBLAS.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindARMPL.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindATLAS.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindCRAY_LIBSCI.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindGenericBLAS.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindNCCL.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindBLIS.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cosma")

install(FILES "${cosma_BINARY_DIR}/cosma.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

if(COSMA_WITH_TESTS)
  add_subdirectory(libs/gtest_mpi)
  enable_testing()
  add_subdirectory(tests)
endif()

if(COSMA_WITH_APPS)
  add_subdirectory(miniapp)
endif()

if(COSMA_WITH_BENCHMARKS AND NOT COSMA_BLAS MATCHES "OPENBLAS")
  add_subdirectory(benchmarks)
endif()
