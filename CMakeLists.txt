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
message(STATUS ${cosma_SOURCE_DIR})
target_include_directories(cosma PUBLIC 
  $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  )
target_compile_features(cosma PUBLIC cxx_std_14)

# Submodules dependencies:
#   grid2grid, options, semiprof, TiledMM
#
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${cosma_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE_CHECK "Check submodules during build" ON)
    if(GIT_SUBMODULE_CHECK)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
        message(STATUS "Updating submodules to the latest commits")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule foreach git pull origin master
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule foreach git pull origin master failed with ${GIT_SUBMOD_RESULT}, \
                please git pull all submodules, to get the latest commits")
        endif()
    endif()
endif()
target_link_libraries(cosma PRIVATE grid2grid semiprof options)

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

# CUDA suppor
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

  install(TARGETS cosma
          EXPORT cosma_targets 
          LIBRARY DESTINATION lib)
          #INCLUDES DESTINATION include/cosma)

  install(EXPORT cosma_targets
          FILE cosmaTargets.cmake
          NAMESPACE cosma::
          DESTINATION lib/cmake/cosma)

  install(DIRECTORY src/cosma 
          DESTINATION include
          FILES_MATCHING PATTERN "*.hpp")

  configure_package_config_file(
    "${cosma_SOURCE_DIR}/cmake/cosmaConfig.cmake.in"
    "${cosma_BINARY_DIR}/cosmaConfig.cmake"
    INSTALL_DESTINATION lib/cmake/cosma)

  write_basic_package_version_file(
    "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
    VERSION ${cosma_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_file("${cosma_SOURCE_DIR}/cmake/cosma.pc.in"
                 "${cosma_BINARY_DIR}/cosma.pc"
                 @ONLY)

  install(FILES "${cosma_BINARY_DIR}/cosmaConfig.cmake"
                "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
          DESTINATION lib/cmake/cosma)
  
  install(FILES ${cosma_BINARY_DIR}/cosma.pc
          DESTINATION lib/pkgconfig/cosma.pc)
  
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
