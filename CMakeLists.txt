cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Check if COSMA is a subproject.
#
set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
  set(MASTER_PROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(cosma VERSION 0.1 LANGUAGES CXX)

# COSMA options
#
option(COSMA_WITH_CUDA "use CUDA GPU backend for base case" OFF)
option(COSMA_WITH_PROFILING "use the semiprof profiler" OFF)
option(COSMA_WITH_TESTS "Generate the test target." ${MASTER_PROJECT})
option(COSMA_WITH_APPS "Generate the miniapp targets." ${MASTER_PROJECT})
option(COSMA_WITH_BENCHMARKS "Generate the benchmark targets." ${MASTER_PROJECT})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

# COSMA library
#
set(cosma_src_prefix "${cosma_SOURCE_DIR}/src/cosma")
set(cosma_src_files 
    "${cosma_src_prefix}/blas.cpp"
    "${cosma_src_prefix}/buffer.cpp"
    "${cosma_src_prefix}/communicator.cpp"
    "${cosma_src_prefix}/context.cpp"
    "${cosma_src_prefix}/interval.cpp"
    "${cosma_src_prefix}/layout.cpp"
    "${cosma_src_prefix}/local_multiply.cpp"
    "${cosma_src_prefix}/mapper.cpp"
    "${cosma_src_prefix}/math_utils.cpp"
    "${cosma_src_prefix}/matrix.cpp"
    "${cosma_src_prefix}/multiply.cpp"
    "${cosma_src_prefix}/one_sided_communicator.cpp"
    "${cosma_src_prefix}/strategy.cpp"
    "${cosma_src_prefix}/two_sided_communicator.cpp"
    )
add_library(cosma STATIC ${cosma_src_files})

target_include_directories(cosma PUBLIC 
    $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>
    )
target_compile_features(cosma PUBLIC cxx_std_14)

# Non-cmake dependencies
#
find_package(MPI REQUIRED)
find_package(MKL REQUIRED)
target_link_libraries(cosma 
    PUBLIC 
    MPI::MPI_CXX 
    PRIVATE
    MKL::MKL
    )
target_compile_definitions(cosma PUBLIC COSMA_WITH_MKL)

set(cosma_submodules_prefix "${PROJECT_SOURCE_DIR}/libs")

# grid2grid dependency
find_package(ScaLAPACK)
if(ScaLAPACK_FOUND)
    add_library(cosma_with_scalapack STATIC
        ${cosma_src_prefix}/scalapack.cpp
        ${cosma_src_prefix}/pgemm.cpp
        )
    target_link_libraries(cosma_with_scalapack PUBLIC cosma MKL::ScaLAPACK)
endif()

# mapper.hpp includes grid2grid.... we need it as a hard dependency otherwise 
# the code fails to compile
#
add_subdirectory("${cosma_submodules_prefix}/grid2grid")
target_link_libraries(cosma PUBLIC grid2grid)

add_subdirectory("${cosma_submodules_prefix}/options")
target_link_libraries(cosma PRIVATE options)

add_subdirectory("${cosma_submodules_prefix}/semiprof")
target_link_libraries(cosma PRIVATE semiprof)


# CUDA gemm back-end
#
if(COSMA_WITH_CUDA)
    add_subdirectory("${cosma_submodules_prefix}/Tiled-MM")
    enable_language(CUDA)
    target_compile_definitions(cosma PRIVATE -DCOSMA_HAVE_GPU)
    target_link_libraries(cosma PUBLIC cublas Tiled-MM)
    target_compile_options(cosma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-arch=sm_60">) 
endif()

# Installation
#
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(TARGETS cosma
    EXPORT cosma_targets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

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


if(COSMA_WITH_PROFILING)
    add_definitions(-DSEMIPROF)
endif()

if(COSMA_WITH_TESTS)
    add_subdirectory("${cosma_submodules_prefix}/gtest_mpi")
    enable_testing()
    add_subdirectory(tests)
endif()

if(COSMA_WITH_APPS)
    add_subdirectory(miniapp)
endif()

if(COSMA_WITH_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
