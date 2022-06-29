cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Check if COSMA is a subproject.
#
set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
    set(MASTER_PROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/build_type.cmake)
include(cmake/adjust_mpiexec_flags.cmake)

# Options
#
option(COSMA_WITH_TESTS "Generate the test target." ${MASTER_PROJECT})
option(COSMA_WITH_APPS "Generate the miniapp targets." ${MASTER_PROJECT})
option(COSMA_WITH_BENCHMARKS "Generate the benchmark targets." ${MASTER_PROJECT})
option(COSMA_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})
option(COSMA_WITH_PROFILING "Enable profiling." OFF)
option(COSMA_WITH_NCCL "Use NCCL as communication backend." OFF)
option(COSMA_WITH_GPU_AWARE_MPI "Use gpu-aware MPI for communication." OFF)
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

set(COSMA_BLAS "MKL" CACHE STRING
    "Blas backend. Can be MKL, OPENBLAS, CRAY_LIBSCI, BLIS, ATLAS, CUDA, ROCM or CUSTOM")
set_property(CACHE COSMA_BLAS PROPERTY STRINGS
    "MKL" "OPENBLAS" "CRAY_LIBSCI" "BLIS" "ATLAS" "CUDA" "ROCM" "CUSTOM")

# the blas library that SCALAPACK is linked to
# this is not used by COSMA, it's just to enable 
# the proper linking to SCALAPACK and its dependencies
set(SCALAPACK_BLAS "OFF" CACHE STRING
    "Blas backend. Can be MKL, OPENBLAS, CRAY_LIBSCI, BLIS, ATLAS, CUSTOM or OFF.")
set_property(CACHE SCALAPACK_BLAS PROPERTY STRINGS
    "OFF" "MKL" "OPENBLAS" "CRAY_LIBSCI" "BLIS" "ATLAS" "CUSTOM")

set(COSMA_SCALAPACK "OFF" CACHE STRING
    "Blas backend. Can be MKL, CRAY_LIBSCI, CUSTOM or OFF.")
set_property(CACHE COSMA_SCALAPACK PROPERTY STRINGS
    "OFF" "MKL" "CRAY_LIBSCI" "CUSTOM")

set(GPU_BACKENDS "CUDA" "ROCM")
if (COSMA_WITH_NCCL AND NOT COSMA_BLAS IN_LIST GPU_BACKENDS) 
    message(FATAL_ERROR "NCCL can only be used with the GPU backend.")
endif()

if (COSMA_WITH_GPU_AWARE_MPI AND NOT COSMA_BLAS IN_LIST GPU_BACKENDS) 
    message(FATAL_ERROR "GPU-aware MPI can only be used with the GPU backend.")
endif()

# if using scalapack, then enforce building as a shared library
# so that the function interpose can work
#if (COSMA_SCALAPACK AND NOT BUILD_SHARED_LIBS)
#    message(FATAL_ERROR "When SCALAPACK wrappers are used, COSMA has to be compiled as a shared library. Rerun cmake adding -DBUILD_SHARED_LIBS=ON.")
# endif()

# if SCALAPACK BLAS is not defined, then set it to be the same
# as the scalapack provider
if (COSMA_SCALAPACK AND NOT SCALAPACK_BLAS) 
    set(SCALAPACK_BLAS ${COSMA_SCALAPACK})
endif()

# check if blas backend is valid
message(STATUS "Selected BLAS backend for COSMA: ${COSMA_BLAS}")
get_property(BACKEND_LIST CACHE COSMA_BLAS PROPERTY STRINGS)
if(NOT COSMA_BLAS IN_LIST BACKEND_LIST)
    message(FATAL_ERROR "Invalid value for COSMA_BLAS!")
endif()

# this does not have to be the case: sometimes we might link COSMA to OPENBLAS, but use NETLIB'S SCALAPACK with NETLIB's blas
# COSMA BLAS and SCALAPACK BLAS can only differ if the GPU version of COSMA is used
# if(NOT ${SCALAPACK_BLAS} STREQUAL ${COSMA_BLAS}) 
#     if((NOT ${COSMA_BLAS} STREQUAL "CUDA") AND (NOT ${COSMA_BLAS} STREQUAL "ROCM"))
#         message(FATAL_ERROR "SCALAPACK must be linked to the same BLAS library as COSMA if the CPU version of COSMA is used!")
#     endif()
# endif()

# check if scalapack backend is valid
message(STATUS "Selected SCALAPACK backend for COSMA: ${COSMA_SCALAPACK}")
unset(BACKEND_LIST)
get_property(BACKEND_LIST CACHE COSMA_SCALAPACK PROPERTY STRINGS)
if(COSMA_SCALAPACK AND NOT COSMA_SCALAPACK IN_LIST BACKEND_LIST)
    message(FATAL_ERROR "Invalid value for COSMA_SCALAPACK!")
endif()

message(STATUS "The BLAS backend that SCALAPACK is linked to: ${SCALAPACK_BLAS}")
unset(BACKEND_LIST)
get_property(BACKEND_LIST CACHE SCALAPACK_BLAS PROPERTY STRINGS)
if(NOT SCALAPACK_BLAS IN_LIST BACKEND_LIST)
    message(FATAL_ERROR "Invalid value for SCALAPACK_BLAS!")
endif()

if (${COSMA_BLAS} STREQUAL "MKL" OR ${COSMA_BLAS} STREQUAL "CRAY_LIBSCI")
    if (COSMA_SCALAPACK AND NOT ${COSMA_BLAS} STREQUAL ${COSMA_SCALAPACK})
        message(FATAL_ERROR "SCALAPACK backend MUST match the BLAS backend if MKL or CRAY_LIBSCI are used!")
    endif()
endif()

project(cosma VERSION 2.6.0 LANGUAGES CXX C)

include(GNUInstallDirs)

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

# Bundled dependencies
#
if (COSMA_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
    add_subdirectory(libs/semiprof)
endif ()

# Do not install cxxopts when including it.
if (COSMA_WITH_TESTS OR COSMA_WITH_APPS)
    option(CXXOPTS_BUILD_EXAMPLES OFF)
    option(CXXOPTS_BUILD_TESTS OFF)
    option(CXXOPTS_ENABLE_INSTALL OFF)
    add_subdirectory(libs/cxxopts)
endif()

# BLAS providers
#
set(BLAS_TARGET "")
set(BLAS_DEF "")
set(GPU OFF)
# BLAS provider that SCALAPACK is linked to
set(SCALAPACK_BLAS_TARGET "")

set(NCCL_TARGET "")
set(NCCL_DEF "")
set(MPI_DEF "")

set(GPU_TARGET "")
set(GPU_DEFF "")

# these are only GPU-backends
if (${COSMA_BLAS} STREQUAL "CUDA" OR ${COSMA_BLAS} STREQUAL "ROCM")
    option(TILEDMM_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
    set(TILEDMM_GPU_BACKEND ${COSMA_BLAS} CACHE STRING "GPU backend" FORCE)
    add_subdirectory(libs/Tiled-MM)
    set(BLAS_TARGET "Tiled-MM")
    set(BLAS_DEF "COSMA_HAVE_GPU")
    set(GPU ON)

    if (COSMA_WITH_NCCL)
        set(NCCL_DEF "COSMA_WITH_NCCL")
        if (${COSMA_BLAS} STREQUAL "CUDA")
            find_package(CUDAToolkit)
            find_package(NCCL)
            set(GPU_TARGET "CUDA::cudart")
            set(NCCL_TARGET "${NCCL_LIBRARIES}")
            message("NCCL INCLUDE DIRS = ${NCCL_INCLUDE_DIRS}")
            message("NCCL LIBRARIES = ${NCCL_LIBRARIES}")
            # set(NCCL_TARGET "${NCCL_LIBRARIES}")
        elseif (${COSMA_BLAS} STREQUAL "ROCM")
            find_package(hip)
            find_package(RCCL)
            set(GPU_TARGET "hip::host")
            set(GPU_DEF "-D__HIP_PLATFORM_HCC__")
            set(NCCL_TARGET "${RCCL_LIBRARIES}")
            message("RCCL INCLUDE DIRS = ${RCCL_INCLUDE_DIRS}")
            message("RCCL LIBRARIES = ${RCCL_LIBRARIES}")
            # set(NCCL_TARGET "${NCCL_LIBRARIES}")
        else()
            message(FATAL_ERROR "COSMA_WITH_NCCL specified, but no GPU backend chosen.")
        endif()
    endif()

    if (COSMA_WITH_GPU_AWARE_MPI)
        set(MPI_DEF "COSMA_WITH_GPU_AWARE_MPI")
        if (${COSMA_BLAS} STREQUAL "CUDA")
            find_package(CUDAToolkit)
            set(GPU_TARGET "CUDA::cudart")
        elseif (${COSMA_BLAS} STREQUAL "ROCM")
            find_package(hip)
            set(GPU_TARGET "hip::host")
            set(GPU_DEF "-D__HIP_PLATFORM_HCC__")
        else()
            message(FATAL_ERROR "COSMA_WITH_GPU_AWARE_MPI specified, but no GPU backend chosen.")
        endif()
    endif()
endif()

# BLAS LIBRARIES below can be used together with the GPU backend
# For this reason, we use MATCHES to also match entries like "MKL,CUDA" for BLAS
if (${COSMA_BLAS} MATCHES "MKL")
    find_package(MKL REQUIRED)
    set(BLAS_TARGET "mkl::mkl_intel_32bit_omp_dyn")
    set(BLAS_DEF "COSMA_WITH_MKL_BLAS")
elseif (${COSMA_BLAS} MATCHES "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(BLAS_TARGET "${CRAY_LIBSCI_LIBRARIES}")
    set(BLAS_DEF "COSMA_WITH_BLAS")
elseif (${COSMA_BLAS} MATCHES "OPENBLAS")
    find_package(OPENBLAS REQUIRED)
    set(BLAS_TARGET "OPENBLAS::openblas")
    set(BLAS_DEF "COSMA_WITH_BLAS")
elseif (${COSMA_BLAS} MATCHES "BLIS")
    find_package(BLIS REQUIRED)
    set(BLAS_TARGET "BLIS::blis")
    set(BLAS_DEF "COSMA_WITH_BLIS_BLAS")
elseif (${COSMA_BLAS} MATCHES "ATLAS")
    find_package(ATLAS REQUIRED)
    set(BLAS_TARGET "ATLAS::atlas")
    set(BLAS_DEF "COSMA_WITH_BLAS")
elseif (${COSMA_BLAS} MATCHES "CUSTOM")
    find_package(BLAS REQUIRED)
    set(BLAS_TARGET "${BLAS_LIBRARIES}")
    set(BLAS_DEF "COSMA_WITH_BLAS")
endif()

# BLAS LIBRARY that SCALAPACK is linked to
if (${SCALAPACK_BLAS} MATCHES "MKL")
    find_package(MKL REQUIRED)
    set(SCALAPACK_BLAS_TARGET "mkl::mkl_intel_32bit_omp_dyn")
elseif (${SCALAPACK_BLAS} MATCHES "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(SCALAPACK_BLAS_TARGET "${CRAY_LIBSCI_LIBRARIES}")
elseif (${SCALAPACK_BLAS} MATCHES "OPENBLAS")
    find_package(OPENBLAS REQUIRED)
    set(SCALAPACK_BLAS_TARGET "OPENBLAS::openblas")
elseif (${SCALAPACK_BLAS} MATCHES "BLIS")
    find_package(BLIS REQUIRED)
    set(SCALAPACK_BLAS_TARGET "BLIS::blis")
elseif (${SCALAPACK_BLAS} MATCHES "ATLAS")
    find_package(ATLAS REQUIRED)
    set(SCALAPACK_BLAS_TARGET "ATLAS::atlas")
elseif (${SCALAPACK_BLAS} MATCHES "CUSTOM")
    find_package(BLAS REQUIRED)
    set(SCALAPACK_BLAS_TARGET "${BLAS_LIBRARIES}")
endif()

set(SCALAPACK_DEPENDENCIES "${SCALAPACK_BLAS_TARGET}")

# (optional) SCALAPACK providers
#
set(SCALAPACK_TARGET "")
if (${COSMA_SCALAPACK} STREQUAL "MKL")
    find_package(MKL REQUIRED)
    set(SCALAPACK_TARGET "mkl::scalapack_${MPI_TYPE}_intel_32bit_omp_dyn")
elseif (${COSMA_SCALAPACK} STREQUAL "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(SCALAPACK_TARGET "${CRAY_LIBSCI_LIBRARIES}")
elseif (${COSMA_SCALAPACK} STREQUAL "CUSTOM")
    find_package(SCALAPACK REQUIRED)
    set(SCALAPACK_TARGET "${SCALAPACK_LIBRARIES}")
else  ()
    message(STATUS "Building with no SCALAPACK interface support.")
endif ()

option(COSTA_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
option(COSTA_WITH_PROFILING "" ${COSMA_WITH_PROFILING})
set(COSTA_SCALAPACK ${COSMA_SCALAPACK} CACHE STRING "")
add_subdirectory(libs/COSTA)

# COSMA
#
add_subdirectory(src/cosma)

if(COSMA_WITH_INSTALL)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

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
endif()

if(COSMA_WITH_TESTS)
    add_subdirectory(libs/gtest_mpi)
    enable_testing()
    add_subdirectory(tests)
endif()

if(COSMA_WITH_APPS)
    add_subdirectory(miniapp)
endif()

if(COSMA_WITH_BENCHMARKS AND NOT COSMA_WITH_OPENBLAS)
    add_subdirectory(benchmarks)

endif()
