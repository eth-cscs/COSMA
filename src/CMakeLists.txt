cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    gpu/tiled_gemm.cpp
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

###################
#  Cosma Library  #
###################
add_library(cosma ${sources})
target_link_libraries(cosma ${cosma_deps})
