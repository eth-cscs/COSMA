cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Check if COSMA is bundled, disable test suit if it is.
#
if(NOT DEFINED PROJECT_NAME)
  set(COSMA_NOT_SUBPROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(cosma VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  "${cosma_SOURCE_DIR}/src/cosma/strategy.cpp"
  "${cosma_SOURCE_DIR}/src/cosma/pdgemm_wrapper.hpp"
  )
message(STATUS ${cosma_SOURCE_DIR})
target_include_directories(cosma PUBLIC ${cosma_SOURCE_DIR}/src)
#target_link_libraries(cosma ${cosma_deps} /opt/cray/dmapp/7.1.1-6.0.7.0_34.3__g5a674e0.ari/lib64/libdmapp.so)

# Submodules dependencies:
#   grid2grid, options, semiprof, TiledMM
#
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${cosma_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE_CHECK "Check submodules during build" ON)
  if(GIT_SUBMODULE_CHECK)
    message(STATUS "Submodule update")
    # execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule foreach git pull origin master
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()
target_link_libraries(cosma PUBLIC grid2grid semiprof options)

# Non-cmake dependencies
#
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)
target_link_libraries(cosma PUBLIC 
  MPI::MPI_CXX 
  OpenMP::OpenMP_CXX 
  MKL::MKL
  )
target_compile_definitions(cosma PUBLIC COSMA_WITH_MKL)

################
#  ScaLAPACK   #
################
if(NOT "${COSMA_SCALAPACK_LIBRARY}" STREQUAL "")
    message("scalapack library = ${COSMA_SCALAPACK_LIBRARY}")
    include(cosma_scalapack)
    cosma_find_scalapack()
    target_link_libraries(cosma ${COSMA_SCALAPACK_LIBRARY})
endif()

################
# CUDA support #
################
option(COSMA_WITH_CUDA "use CUDA GPU backend for base case" OFF)
if(COSMA_WITH_CUDA)
    enable_language(CUDA)
    target_compile_definitions(cosma PRIVATE -DCOSMA_HAVE_GPU)
    target_link_libraries(cosma PRIVATE cublas Tiled-MM)
    target_compile_options(cosma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:"-arch=sm_60">) 
endif()

#################################
#          PROFILER             #
#################################
option(COSMA_WITH_PROFILING "use the semiprof profiler" OFF)
if(COSMA_WITH_PROFILING)
    add_definitions(-DSEMIPROF)
endif()

#############
#  Testing  #
#############
if(COSMA_NOT_SUBPROJECT)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

#####################################
#  Adding various code directories  #
#####################################
add_subdirectory(libs)
add_subdirectory(miniapp)
add_subdirectory(benchmarks)
