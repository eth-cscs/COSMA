cmake_minimum_required(VERSION 3.5)

project(carma-mm)
set(PROJECT_NAME carma-mm)

include(./cmake/utils.cmake)

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

#############
# Cilk Part #
#############
set(cilk_deps cilkrts)
##############
# /Cilk Part #
##############

#################################
#  Generic Compilation options  #
#################################

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#Compiler must support c++14
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CXX_FLAGS -std=c++14 -fcilkplus)
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

############
# MKL Part #
############
if (USE_BLAS_DEFAULT)
    FIND_PACKAGE(BLAS REQUIRED)
    find_path(BLAS_INCLUDE_DIRS cblas.h
        /usr/include
        /usr/local
        /usr/local/include
        $ENV{BLAS_HOME}/include)
    include_directories(${BLAS_INCLUDE_DIRS})
    if (NOT((HOSTNAME MATCHES ".*daint.*") OR (HOSTNAME MATCHES "tave*")))
        set(blas_deps blas)
    endif ()
elseif (USE_BLAS_MKL)
    set(MKL_MULTI_THREADED ON)
    FIND_PACKAGE(MKL)
    if (MKL_FOUND)
        add_definitions(-DUSE_MKL)
        include_directories(${MKL_INCLUDE_DIR})
        link_directories(${MKL_LIBRARY_PATH})
        set(blas_deps ${MKL_LIBRARIES})
    endif()
endif ()
#############
# /MKL Part #
#############

##############
#  Cppcheck  #
##############

# Add a target to generate Cppcheck report
include(${PRJ_CMAKE_DIR}/gen_cppcheck.cmake)
GENERATE_CPPCHECK(SOURCEDIRS src tests
    ENABLE_IDS warning style performance portability information
    INCLUDES ${CURRENT_SOURCE_DIR}
    PROJECT_NAME ${PROJECT_NAME}
    )

############
# TBB Part #
############
find_package(TBB)
if (TBB_FOUND)
    include_directories ( ${TBB_INCLUDE_DIR} )
    link_directories ( ${TBB_LIBRARY_DIR} )
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(tbb_deps ${_tbb_deps_debug})
    else ()
        set(tbb_deps ${_tbb_deps})
    endif()
endif ()
#############
# /TBB Part #
#############

#############
#  Testing  #
#############
enable_testing()

#####################################
#  Adding various code directories  #
#####################################

add_subdirectory(src)
if (MAKE_TEST)
    add_subdirectory(tests)
endif()
