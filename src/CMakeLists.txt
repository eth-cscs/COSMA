cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    cmd_parser.cpp
    communication.cpp
    carma.cpp
    matrix.cpp
    layout.cpp
    mapper.cpp
    statistics.hpp
)

###################
#  Carma Library  #
###################
add_library(carma ${sources})
target_link_libraries(carma semiprof)

