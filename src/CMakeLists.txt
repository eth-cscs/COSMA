cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    gpu/gemm.cpp
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
    options.cpp
    math_utils.cpp
)

set (cuda_sources 
    #gpu/gemm.cu
    #gpu/cuda_event.hpp
    #gpu/cuda_stream.hpp
    #gpu/device_vector.cu
)

###################
#  Cosma Library  #
###################

add_library(cosma ${sources})

if(COSMA_WITH_CUDA)
    target_link_libraries(cosma semiprof ${mpi_deps} cublas)
else()
    target_link_libraries(cosma semiprof ${mpi_deps})
endif()

