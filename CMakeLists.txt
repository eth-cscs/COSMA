cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
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
target_link_libraries(cosma ${cosma_deps})
#target_link_libraries(cosma ${cosma_deps} /opt/cray/dmapp/7.1.1-6.0.7.0_34.3__g5a674e0.ari/lib64/libdmapp.so)
# target_include_directories(cosma PUBLIC ${MPI_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src)



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


############
# MPI Part #
############
find_package(MPI REQUIRED)
if (MPI_FOUND)
    add_definitions(-DUSE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
    #runtime libs for mpi
    set(mpi_deps ${MPI_CXX_LIBRARIES})
    option(REMOTE "--oversubscribe flag for OpenMPI" OFF)
    set(OVERSUBSCRIBE "--oversubscribe")
    if(REMOTE)
        set(OVERSUBSCRIBE "")
    endif()
endif ()

###############
# OpenMP Part #
###############
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

################
# BLAS Part    #
################
find_package(BLAS REQUIRED)
set(blas_deps ${BLAS_LIBRARIES})
#include(cosma_lapack)
#cosma_find_lapack()
#set(blas_deps ${COSMA_LAPACK_LIBRARY})

################
#  ScaLAPACK   #
################
if(NOT "${COSMA_SCALAPACK_LIBRARY}" STREQUAL "")
    message("scalapack library = ${COSMA_SCALAPACK_LIBRARY}")
    include(cosma_scalapack)
    cosma_find_scalapack()
    set(scalapack_deps ${COSMA_SCALAPACK_LIBRARY})
endif()

################
# CUDA support #
################
option(COSMA_WITH_CUDA "use CUDA GPU backend for base case" OFF)
if(COSMA_WITH_CUDA)
    #enable_language(CUDA)
    #set(CMAKE_CXX_STANDARD 11)
    find_package(CUDA REQUIRED)
    add_definitions(-DCOSMA_HAVE_GPU)
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60;-std=c++11;-Xcompiler -fPIC)
    # list(APPEND CMAKE_CUDA_FLAGS -gencode arch=compute_60,code=sm_60)
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60 -gencode=arch=compute_60,code=sm_60")
    set(cuda_deps ${CUDA_LIBRARIES} cublas Tiled-MM)
else()
    set(cuda_deps "")
endif()

set(cosma_deps ${mpi_deps} ${blas_deps} options ${cuda_deps} semiprof grid2grid)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")




#############
#  Testing  #
#############
include(CTest)
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
add_subdirectory(libs)
add_subdirectory(src)
add_subdirectory(miniapp)
add_subdirectory(tests)
