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
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

set(COSMA_BLAS "MKL" CACHE STRING
    "Blas backend. Can be MKL, OPENBLAS, CRAY_LIBSCI, CUSTOM, CUDA or ROCM.")
set_property(CACHE COSMA_BLAS PROPERTY STRINGS
    "MKL" "OPENBLAS" "CRAY_LIBSCI" "CUSTOM" "CUDA" "ROCM")

set(COSMA_SCALAPACK "OFF" CACHE STRING
    "Blas backend. Can be MKL, CRAY_LIBSCI, CUSTOM or OFF.")
set_property(CACHE COSMA_SCALAPACK PROPERTY STRINGS
    "OFF" "MKL" "CRAY_LIBSCI" "CUSTOM")

# check if blas backend is valid
message(STATUS "Selected BLAS backend for COSMA: ${COSMA_BLAS}")
get_property(BACKEND_LIST CACHE COSMA_BLAS PROPERTY STRINGS)
if(NOT COSMA_BLAS IN_LIST BACKEND_LIST)
  message(FATAL_ERROR "Invalid value for COSMA_BLAS!")
endif()

# check if scalapack backend is valid
message(STATUS "Selected ScaLAPACK backend for COSMA: ${COSMA_SCALAPACK}")
unset(BACKEND_LIST)
get_property(BACKEND_LIST CACHE COSMA_SCALAPACK PROPERTY STRINGS)
if(COSMA_SCALAPACK AND NOT COSMA_SCALAPACK IN_LIST BACKEND_LIST)
  message(FATAL_ERROR "Invalid value for COSMA_SCALAPACK!")
endif()

if (NOT ${COSMA_BLAS} STREQUAL "CUDA" AND NOT ${COSMA_BLAS} STREQUAL "ROCM" AND NOT ${COSMA_BLAS} STREQUAL "OPENBLAS")
  if (COSMA_SCALAPACK AND NOT ${COSMA_BLAS} STREQUAL ${COSMA_SCALAPACK})
      message(FATAL_ERROR "ScaLAPACK backend MUST match BLAS backend if no GPU backend is used!")
  endif()
endif()

project(cosma VERSION 2.2.0 LANGUAGES CXX)

# preserve rpaths when installing and make the install folder relocatable
# use `CMAKE_SKIP_INSTALL_RPATH` to skip this
# https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES 
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
# skip RPATH if SIRIUS is installed to system directories
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
#
find_package(MPI REQUIRED)
adjust_mpiexec_flags()

# Bundled dependencies
#
if (COSMA_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
    add_subdirectory(libs/semiprof)
endif ()

option(OPTIONS_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
add_subdirectory(libs/cxxopts)

option(GRID2GRID_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
option(GRID2GRID_WITH_PROFILING "" ${COSMA_WITH_PROFILING})
add_subdirectory(libs/grid2grid)

# BLAS providers
#
set(BLAS_TARGET "")
set(BLAS_DEF "")

if (${COSMA_BLAS} STREQUAL "MKL")
    find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP)
    set(BLAS_TARGET "mkl::blas_32bit_omp")
    set(BLAS_DEF "COSMA_WITH_MKL_BLAS")
elseif (${COSMA_BLAS} STREQUAL "CUDA" OR ${COSMA_BLAS} STREQUAL "ROCM")
    option(TILEDMM_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
    set(TILEMM_GPU_BACKEND ${COSMA_BLAS} CACHE STRING FORCE "")
    add_subdirectory(libs/Tiled-MM)
    set(BLAS_TARGET "Tiled-MM")
    set(BLAS_DEF "COSMA_HAVE_GPU")
elseif (${COSMA_BLAS} STREQUAL "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    set(BLAS_TARGET "openblas")
    set(BLAS_DEF "COSMA_WITH_BLAS")
elseif (${COSMA_BLAS} STREQUAL "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(BLAS_TARGET "${CRAY_LIBSCI_LIBRARIES}")
    set(BLAS_DEF "COSMA_WITH_BLAS")
elseif (${COSMA_BLAS} STREQUAL "CUSTOM")
    find_package(BLAS REQUIRED)
    set(BLAS_TARGET "${BLAS_LIBRARIES}")
    set(BLAS_DEF "COSMA_WITH_BLAS")
endif ()

# (optional) ScaLAPACK providers
#
set(ScaLAPACK_TARGET "")
if (${COSMA_SCALAPACK} STREQUAL "MKL")
    find_package(MKL REQUIRED COMPONENTS SCALAPACK_32BIT_OMP)
    set(ScaLAPACK_TARGET "mkl::scalapack_32bit_omp")
elseif (${COSMA_SCALAPACK} STREQUAL "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(ScaLAPACK_TARGET "${CRAY_LIBSCI_LIBRARIES}")
elseif (${COSMA_SCALAPACK} STREQUAL "CUSTOM")
    find_package(SCALAPACK REQUIRED)
    set(ScaLAPACK_TARGET "${SCALAPACK_LIBRARIES}")
else  ()
    message(STATUS "Building with no ScaLAPACK interface support.")
endif ()

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
