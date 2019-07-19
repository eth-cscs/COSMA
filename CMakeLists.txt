cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Check if COSMA is bundled, disable test suit if it is.
#
if(NOT DEFINED PROJECT_NAME)
    set(COSMA_NOT_SUBPROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(cosma VERSION 0.1 LANGUAGES CXX)

# COSMA options
#
option(COSMA_WITH_CUDA "use CUDA GPU backend for base case" OFF)
option(COSMA_WITH_PROFILING "use the semiprof profiler" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

# COSMA library
#
add_library(cosma STATIC
    "${cosma_SOURCE_DIR}/src/cosma/blas.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/scalapack.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/pgemm.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/buffer.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/communicator.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/context.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/interval.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/layout.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/local_multiply.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/mapper.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/math_utils.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/matrix.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/multiply.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/one_sided_communicator.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/strategy.cpp"
    "${cosma_SOURCE_DIR}/src/cosma/two_sided_communicator.cpp"
    )
target_include_directories(cosma PUBLIC 
  $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>
  )
target_compile_features(cosma PUBLIC cxx_std_14)

# Non-cmake dependencies
#
find_package(grid2grid REQUIRED)
find_package(semiprof REQUIRED)
find_package(options REQUIRED)
find_package(MPI REQUIRED)
find_package(MKL REQUIRED)
target_link_libraries(cosma 
    PUBLIC 
    MPI::MPI_CXX 
    PRIVATE
    MKL::MKL
    grid2grid::grid2grid 
    semiprof::semiprof 
    options::options
  )
target_compile_definitions(cosma PUBLIC COSMA_WITH_MKL)

find_package(ScaLAPACK)
if(ScaLAPACK_FOUND)
  # TODO
endif()

# CUDA support
#
if(COSMA_WITH_CUDA)
    enable_language(CUDA)
    target_compile_definitions(cosma PRIVATE -DCOSMA_HAVE_GPU)
    target_link_libraries(cosma PRIVATE cublas Tiled-MM)
    target_compile_options(cosma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-arch=sm_60">) 
endif()

# PROFILER
#
if(COSMA_WITH_PROFILING)
    add_definitions(-DSEMIPROF)
endif()

# Installation
#
if (COSMA_NOT_SUBPROJECT)
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
endif()

# Testing
#
if(COSMA_NOT_SUBPROJECT)
    enable_testing()
    include(CTest)
    add_subdirectory(tests)
endif()

add_subdirectory(libs)
add_subdirectory(miniapp)
add_subdirectory(benchmarks)
