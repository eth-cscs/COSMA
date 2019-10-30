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

if (COSMA_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ON)
endif()

option(COSMA_WITH_GPU "Use CUDA cuBLAS GPU back end." OFF)
option(COSMA_WITH_MKL_BLAS "Use MKL BLAS back end." OFF)
option(COSMA_WITH_OPENBLAS "Use OpenBLAS backend." OFF)
option(COSMA_WITH_NETLIB_BLAS "Use Netlib BLAS backend." OFF)

option(COSMA_WITH_MKL_ScaLAPACK "Use MKL ScaLAPACK back end." OFF) 
option(COSMA_WITH_NETLIB_ScaLAPACK "Use Netlib ScaLAPACK back end." OFF) 

option(COSMA_WITH_FOTRAN_BINDINGS "Make Fortran bindings to COSMA." OFF) 

project(cosma VERSION 0.1 LANGUAGES CXX)


# Dependencies
#
find_package(MPI REQUIRED)
adjust_mpiexec_flags()

# Bundled dependencies
#

if (COSMA_WITH_PROFILING)
    add_subdirectory(libs/semiprof)
endif ()

option(OPTIONS_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
add_subdirectory(libs/options)

option(GRID2GRID_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
option(GRID2GRID_WITH_PROFILING "" ${COSMA_WITH_PROFILING})
add_subdirectory(libs/grid2grid)

# BLAS providers
#
set(BLAS_TARGET "")
set(BLAS_DEF "")
if (COSMA_WITH_MKL_BLAS)
    find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP)
    set(BLAS_TARGET "mkl::blas_32bit_omp")
    set(BLAS_DEF "COSMA_WITH_MKL_BLAS")
elseif (COSMA_WITH_GPU)
    option(TILEDMM_WITH_INSTALL "" ${COSMA_WITH_INSTALL}) 
    add_subdirectory(libs/Tiled-MM)
    set(BLAS_TARGET "Tiled-MM")
    set(BLAS_DEF "COSMA_HAVE_GPU")
elseif (COSMA_WITH_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    add_library(openblas SHARED IMPORTED)
    set_target_properties(openblas PROPERTIES 
                                   INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
                                   IMPORTED_LOCATION "${OpenBLAS_LIBRARIES}")
    set(BLAS_TARGET "openblas")
    set(BLAS_DEF "COSMA_WITH_BLAS")
elseif (COSMA_WITH_NETLIB_BLAS)
    find_package(cblas NO_MODULE REQUIRED)
    set(BLAS_TARGET "cblas")
    set(BLAS_DEF "COSMA_WITH_BLAS")
else ()
    message(FATAL_ERROR "No BLAS back end found! Select a back end and make sure it's installed with support for the cblas interface and CMake config files.")
endif ()

# (optional) ScaLAPACK providers 
#
set(ScaLAPACK_TARGET "")
if (COSMA_WITH_MKL_ScaLAPACK)
    find_package(MKL REQUIRED COMPONENTS SCALAPACK_32BIT_OMP)
    set(ScaLAPACK_TARGET "mkl::scalapack_32bit_omp")
elseif (COSMA_WITH_NETLIB_ScaLAPACK) 
    find_package(scalapack REQUIRED)  
    set(ScaLAPACK_TARGET "scalapack")
else  ()
    message(STATUS "Building with no ScaLAPACK interface support.")
endif ()

# COSMA
#
add_subdirectory(src/cosma)


if(COSMA_WITH_INSTALL)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(EXPORT cosma_targets
            FILE cosmaTargets.cmake
            NAMESPACE cosma::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cosma")

    install(DIRECTORY "${cosma_SOURCE_DIR}/src/cosma" 
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.hpp")

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
