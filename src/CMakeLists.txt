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
    mpi_allocator.hpp
    scheduleGenerator.cpp
)

set (cuda_sources 
    gpu/gemm.cu
    gpu/cuda_event.hpp
    gpu/cuda_stream.hpp
)

###################
#  Cosma Library  #
###################
add_library(cosma ${sources})
target_link_libraries(cosma semiprof ${mpi_deps})

if(COSMA_WITH_CUDA)
    cuda_add_library(cosmacu ${cuda_sources})
    target_link_libraries(cosmacu cublas)
endif()

