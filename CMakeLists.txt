cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
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
add_library(cosma "")
target_include_directories(cosma PUBLIC ${cosma_SOURCE_DIR}/src)
#target_link_libraries(cosma ${cosma_deps} /opt/cray/dmapp/7.1.1-6.0.7.0_34.3__g5a674e0.ari/lib64/libdmapp.so)

#################################
#  Generic Compilation options  #
#################################
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lstdc++")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -pthread")
endif()


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
  endif()
endif()
target_link_libraries(cosma PRIVATE options semiprof grid2grid)

# Non-cmake dependencies
#
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

target_compile_definitions(cosma PRIVATE -DUSE_MPI)
target_link_libraries(cosma PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${BLAS_LIBRARIES})

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

#################################
#        COVERAGE TEST          #
#################################
option(COSMA_COVERAGE_TEST "use lcov for the code coverage test" OFF)
if(COSMA_COVERAGE_TEST)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} --coverage")

    add_custom_target("coverage"
                      COMMAND lcov --capture --directory ${CMAKE_CURRENT_BINARY_DIR} --output-file coverage.info
                      COMMAND genhtml coverage.info --output-directory ${CMAKE_CURRENT_BINARY_DIR}/coverage
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  else ()
    message(WARNING "Coverage test can be used only with GCC.")
    set(COSMA_COVERAGE_TEST "OFF" FORCE)
  endif()
endif()

#############
#  Testing  #
#############
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
add_subdirectory(libs)
add_subdirectory(src)
add_subdirectory(miniapp)
add_subdirectory(tests)
