cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
include(GNUInstallDirs)

# Check if COSMA is a subproject.
#
set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
  set(MASTER_PROJECT ON)
endif()

project(cosma
  DESCRIPTION "Communication Optimal Matrix Multiplication"
  HOMEPAGE_URL "https://github.com/eth-cscs/COSMA"
  VERSION 2.6.2
  LANGUAGES CXX C)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(cmake/build_type.cmake)
include(cmake/adjust_mpiexec_flags.cmake)
include(GitSubmodule)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES") # always write compile_commands.json

# Options
#

set(COSMA_GPU_BACKENDS_LIST "CUDA" "ROCM")
set(COSMA_SCALAPACK_LIST "OFF" "MKL" "CRAY_LIBSCI" "CUSTOM")
set(COSMA_BLAS_LIST   "auto" "MKL" "OPENBLAS" "CRAY_LIBSCI" "CUSTOM" "BLIS" "ATLAS" "CUDA" "ROCM" "NONE")
option(COSMA_WITH_TESTS "Generate the test target." ${MASTER_PROJECT})
option(COSMA_WITH_APPS "Generate the miniapp targets." ${MASTER_PROJECT})
option(COSMA_WITH_BENCHMARKS "Generate the benchmark targets." ${MASTER_PROJECT})
option(COSMA_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})
option(COSMA_WITH_PROFILING "Enable profiling." OFF)
option(COSMA_WITH_NCCL "Use NCCL as communication backend." OFF)
option(COSMA_WITH_GPU_AWARE_MPI "Use gpu-aware MPI for communication." OFF)
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)
set(COSMA_SCALAPACK "OFF" CACHE STRING "scalapack implementation. Can be MKL, CRAY_LIBSCI, CUSTOM or OFF.")
set(COSMA_BLAS "NONE" CACHE STRING "Blas library for computations on host or GPU")

set(COSMA_BLAS_VENDOR "NONE")
set(COSMA_GPU_BACKEND "NONE")

set_property(CACHE COSMA_SCALAPACK PROPERTY STRINGS ${COSMA_SCALAPACK_LIST})
set_property(CACHE COSMA_BLAS PROPERTY STRINGS ${COSMA_BLAS_LIST})

if (COSMA_BLAS MATCHES "CUDA|ROCM")
  set(COSMA_GPU_BACKEND ${COSMA_BLAS})
else()
  if(COSMA_BLAS STREQUAL "NONE")
    message("A Blas implementation is needed when running on CPU only: choices are : auto, MKL, OPENBLAS, CRAY_LIBSCI, CUSTOM, BLIS, ATLAS")
  else()
    set(COSMA_BLAS_VENDOR ${COSMA_BLAS})
  endif()
endif()

if (COSMA_WITH_NCCL AND NOT COSMA_GPU_BACKEND IN_LIST COSMA_GPU_BACKENDS_LIST)
  message(FATAL_ERROR "NCCL can only be used with the GPU backend.")
endif()

if (COSMA_WITH_GPU_AWARE_MPI AND NOT COSMA_GPU_BACKEND IN_LIST COSMA_GPU_BACKENDS_LIST)
  message(FATAL_ERROR "GPU-aware MPI can only be used with the GPU backend.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()


# check if scalapack backend is valid
message(STATUS "Selected SCALAPACK backend for COSMA: ${COSMA_SCALAPACK}")
if(NOT COSMA_SCALAPACK IN_LIST COSMA_SCALAPACK_LIST)
  message(FATAL_ERROR "Invalid value for COSMA_SCALAPACK!")
endif()

# the blas targets are only defined when COSMA_SCALAPACK is ON whatever value of COSMA_GPU_BACKEND 
if (COSMA_SCALAPACK OR COSMA_GPU_BACKEND MATCHES "OFF")
  if (COSMA_SCALAPACK MATCHES "MKL" OR COSMA_SCALAPACK MATCHES "CRAY_LIBSCI")
    set(COSMA_BLAS_VENDOR ${COSMA_SCALAPACK})
  endif()
endif()

if (NOT COSMA_BLAS_VENDOR STREQUAL "NONE")
  find_package(Blas REQUIRED)
endif()

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

# Dependencies
# MPI
find_package(MPI REQUIRED)
adjust_mpiexec_flags()

if (NOT COSMA_SCALAPACK MATCHES "OFF")
  find_package(SCALAPACK REQUIRED)
endif()

# Bundled dependencies
#
if (COSMA_WITH_PROFILING)
  option(SEMIPROF_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
  find_package(semiprof REQUIRED)
endif ()

# Do not install cxxopts when including it.
if (COSMA_WITH_TESTS OR COSMA_WITH_APPS)
  find_package(cxxopts REQUIRED)
endif()

# these are only GPU-backends
if (COSMA_GPU_BACKEND MATCHES "CUDA|ROCM")
  set(TILEDMM_GPU_BACKEND ${COSMA_GPU_BACKEND} CACHE STRING "GPU backend" FORCE)
  add_git_submodule_or_find_external(Tiled-MM libs/Tiled-MM)
  if (NOT TARGET Tiled-MM::Tiled-MM AND TARGET Tiled-MM)
    add_library(Tiled-MM::Tiled-MM ALIAS Tiled-MM)
    if (TARGET Tiled-MM)
      message(STATUS "Test 2")
    endif()
  endif()

  if (COSMA_WITH_NCCL OR COSMA_WITH_GPU_AWARE_MPI)
    if (${COSMA_GPU_BACKEND} MATCHES "CUDA")
      find_package(CUDAToolkit REQUIRED)
      find_package(NCCL REQUIRED)
      message("NCCL INCLUDE DIRS = ${NCCL_INCLUDE_DIRS}")
      message("NCCL LIBRARIES = ${NCCL_LIBRARIES}")
    elseif(${COSMA_GPU_BACKEND} MATCHES "ROCM")
      finmd_package(hip REQUIRED)
      find_package(RCCL REQUIRED)
      message("RCCL INCLUDE DIRS = ${RCCL_INCLUDE_DIRS}")
      message("RCCL LIBRARIES = ${RCCL_LIBRARIES}")
    else()
      message(FATAL_ERROR "COSMA_WITH_NCCL AND/OR COSMA_WITH_GPU_AWARE_MPI are specified, but no GPU backend chosen.")
    endif()
  endif()
endif()

set(COSTA_WITH_PROFILING "" ${COSMA_WITH_PROFILING} CACHE STRING "" FORCE)
set(COSTA_SCALAPACK ${COSMA_SCALAPACK} CACHE STRING "" FORCE)

if (TARGET cosma::scalapack::scalapack AND NOT COSMA_SCALAPACK MATCHES "OFF")
  add_library(costa::scalapack::scalapack ALIAS cosma::scalapack::scalapack)
endif()

add_git_submodule_or_find_external(costa libs/COSTA)

if (NOT TARGET costa::costa)  # alias targets for add_subdirectory dependency
  add_library(costa::costa ALIAS costa)
endif()

if (TARGET costa_prefixed_scalapack AND (NOT TARGET costa::costa_prefixed_scalapack))
  add_library(costa::costa_prefixed_scalapack ALIAS costa_prefixed_scalapack)
  add_library(costa::costa_scalapack ALIAS costa_scalapack)
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

install(FILES "${cosma_BINARY_DIR}/cosmaConfig.cmake"
  "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
  "${cosma_SOURCE_DIR}/cmake/FindMKL.cmake"
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
