cmake_minimum_required(VERSION 3.5)
project(carma)

set(PROJECT_NAME carma CXX C)
set(CMAKE_CXX_STANDARD 14)

include(./cmake/utils.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

#Destination
if (BINDIR)
    set(RUNTIME_DEST ${BINDIR})
else ()
    set(RUNTIME_DEST ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif ()

#cmake refs
site_name(HOSTNAME)
set(PRJ_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH "${PRJ_CMAKE_DIR}")

#local ref
set(PRJ_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#################################
#          PROFILER             #
#################################
option(CARMA_WITH_PROFILING "use the semiprof profiler" OFF)
if(CARMA_WITH_PROFILING)
    add_definitions(-DSEMIPROF)
endif()

include_directories("${PRJ_ROOT_DIR}/profiler")
include_directories("${PRJ_ROOT_DIR}/profiler/include/semiprof")

#################################
#        COVERAGE TEST          #
#################################
option(CARMA_COVERAGE_TEST "use lcov for the code coverage test" OFF)
if(CARMA_COVERAGE_TEST)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} --coverage")

    add_custom_target("coverage"
                      COMMAND lcov --capture --directory ${CMAKE_CURRENT_BINARY_DIR} --output-file coverage.info
                      COMMAND genhtml coverage.info --output-directory ${CMAKE_CURRENT_BINARY_DIR}/coverage
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  else ()
    message(WARNING "Coverage test can be used only with GCC.")
    set(CARMA_COVERAGE_TEST "OFF" FORCE)
  endif()
endif()


#################################
#  Generic Compilation options  #
#################################
#Compiler must support c++14
if (CMAKE_COMPILER_IS_GNUCXX)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CXX_FLAGS ${CXX_FLAGS} -Wall -O0 -ggdb -DDEBUG)
    else ()
        set(CXX_FLAGS ${CXX_FLAGS} -O3)
    endif ()
    set(CXX_FLAGS ${CXX_FLAGS} -fopenmp)
    add_compile_options(${CXX_FLAGS})
endif ()

############
# MPI Part #
############
find_package(MPI)
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
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

################
# BLAS Part    #
################
include(lapack)
carma_find_lapack()
set(blas_deps ${CARMA_LAPACK_LIBRARY})

################
# CUDA support #
################
option(CARMA_WITH_CUDA "use CUDA GPU backend for base case" OFF)
if(CARMA_WITH_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DCARMA_HAVE_GPU)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60;-std=c++11;-Xcompiler -fPIC)
    set(cuda_deps ${CUDA_LIBRARIES})
else()
    set(cuda_deps "")
endif()

set(carma_deps ${mpi_deps} ${blas_deps} ${cuda_deps})

#############
#  Testing  #
#############
include(CTest)
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(miniapp)
add_subdirectory(profiler)

