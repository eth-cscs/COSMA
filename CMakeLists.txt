cmake_minimum_required(VERSION 3.5)
project(cosma)

set(PROJECT_NAME cosma CXX C)
set(CMAKE_CXX_STANDARD 14)

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

#################################
#          PROFILER             #
#################################
option(COSMA_WITH_PROFILING "use the semiprof profiler" OFF)
if(COSMA_WITH_PROFILING)
    add_definitions(-DSEMIPROF)
    set(prof_deps semiprof)
else()
    set(prof_deps "")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/profiler")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/profiler/include/semiprof")

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
#Compiler must support c++14
if (CMAKE_COMPILER_IS_GNUCXX)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CXX_FLAGS ${CXX_FLAGS} -Wall -O0 -DDEBUG)
        # add address sanitizer for debugging purposes
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
    else ()
        set(CXX_FLAGS ${CXX_FLAGS} -O3)
    endif ()
    add_compile_options(${CXX_FLAGS})
endif ()

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
include(lapack)
cosma_find_lapack()
set(blas_deps ${COSMA_LAPACK_LIBRARY})

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
    #list(APPEND CMAKE_CUDA_FLAGS -arch=sm_60)
    set(cuda_deps ${CUDA_LIBRARIES} cublas)
else()
    set(cuda_deps "")
endif()

set(cosma_deps ${mpi_deps} ${blas_deps} ${cuda_deps} ${prof_deps})
message("cuda deps: ${cuda_deps}")

#############
#  Testing  #
#############
include(CTest)
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(miniapp)
add_subdirectory(profiler)

