cmake_minimum_required(VERSION 3.5)

project(carma)
set(PROJECT_NAME carma CXX C)
set(CMAKE_CXX_STANDARD 14)

include(./cmake/utils.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

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

#----------------------------------------------------------
# Built-in profiler
#----------------------------------------------------------
option(CARMA_WITH_PROFILING "use built-in profiling" OFF)
if(CARMA_WITH_PROFILING)
    add_definitions(-DCARMA_HAVE_PROFILING)
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
        #if (OPENMP_FOUND)
        #set(CXX_FLAGS ${CXX_FLAGS} ${OpenMP_CXX_FLAGS})
        #set(openmp_deps gomp)
        #endif ()
    endif ()
    add_compile_options(${CXX_FLAGS})
endif ()

##############################
#  Cuda Compilation options  #
##############################
if (USE_CUDA)
    find_package(CUDA 8.0)
    if (CUDA_FOUND)
        add_definitions(-DUSE_CUDA)

        if (THRUST_BACKEND STREQUAL "OMP")
            list(APPEND CUDA_NVCC_FLAGS
                -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
                -Xcompiler -fopenmp)
            list(APPEND cuda_deps gomp)
        else () #(THRUST_BACKEND STREQUAL "CUDA")
            list(APPEND CUDA_NVCC_FLAGS
                -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
        endif ()

        #Eventually add some code instrumentation
        if (USE_NVTX)
            add_definitions(-DUSE_NVTX)
        endif ()

        #generic compilation flag, defines target architecture, type of build, ...
        list(APPEND CUDA_NVCC_FLAGS
            -gencode arch=compute_60,code=sm_60
            -std=c++11
            --expt-extended-lambda
            -rdc=true
            -Xcompiler -fPIC)

        set(CUDA_SEPARABLE_COMPILATION ON)

        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            list(APPEND CUDA_NVCC_FLAGS
                --device-debug
                --generate-line-info
                -g
                -G
                -DTHRUST_DEBUG )
        else ()
            list(APPEND CUDA_NVCC_FLAGS
                -use_fast_math
                -O3)
        endif ()
        #runtime libs for cuda
        list(APPEND cuda_deps cudadevrt nvToolsExt)
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    endif ()
endif ()
###############################
#  /Cuda Compilation options  #
###############################

############
# MPI Part #
############
find_package(MPI)
if (MPI_FOUND)
    add_definitions(-DUSE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
    #runtime libs for mpi
    set(mpi_deps ${MPI_CXX_LIBRARIES})
endif ()
#############
# /MPI Part #
#############

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
# /OpenMP Part #
################

################
# BLAS Part    #
################
include(lapack)
carma_find_lapack()
set(blas_deps ${CARMA_LAPACK_LIBRARY})
################
# /BLAS Part   #
################

##############
#  Cppcheck  #
##############
# Add a target to generate Cppcheck report
include(${PRJ_CMAKE_DIR}/gen_cppcheck.cmake)
GENERATE_CPPCHECK(SOURCEDIRS src tests
    ENABLE_IDS warning style performance portability information
    INCLUDES ${CURRENT_SOURCE_DIR}
    PROJECT_NAME ${PROJECT_NAME})
##############
# /Cppcheck  #
##############

set(carma_deps ${mpi_deps} ${blas_deps})

#############
#  Testing  #
#############
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
add_subdirectory(profiler)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(miniapp)

