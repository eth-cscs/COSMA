cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    communicator.cpp
    multiply.cpp
    matrix.cpp
    layout.cpp
    mapper.cpp
    statistics.hpp
    strategy.cpp
    interval.cpp
    timer.hpp
    buffer.cpp
    allocator.hpp
)

set (cuda_sources 
    gpu/gemm.cu
    gpu/cuda_event.hpp
    gpu/cuda_stream.hpp
)

###################
#  Carma Library  #
###################
add_library(carma ${sources})
target_link_libraries(carma semiprof ${mpi_deps})

if(CARMA_WITH_CUDA)
    cuda_add_library(carmacu ${cuda_sources})
    target_link_libraries(carmacu cublas)
endif()

