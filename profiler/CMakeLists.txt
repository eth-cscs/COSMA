cmake_minimum_required(VERSION 3.5)

###############
#  Src files  #
###############

set(sources
    profiler.cpp
)

######################
#  Profiler Library  #
######################

add_library(profiler ${sources})

