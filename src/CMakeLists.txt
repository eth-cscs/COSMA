cmake_minimum_required(VERSION 3.5)

#include_directories("${PRJ_ROOT_DIR}/profiler")

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
    profiler.cpp
    statistics.hpp
)

###################
#  Carma Library  #
###################
add_library(carma ${sources})
#target_link_libraries(carma profiler)

