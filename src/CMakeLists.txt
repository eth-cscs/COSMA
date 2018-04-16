cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    communicator.cpp
    carma.cpp
    matrix.cpp
    layout.cpp
    mapper.cpp
    statistics.hpp
    strategy.cpp
    topology.cpp
)

###################
#  Carma Library  #
###################
add_library(carma ${sources})
target_link_libraries(carma semiprof)

