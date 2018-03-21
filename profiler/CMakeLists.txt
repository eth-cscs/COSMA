cmake_minimum_required(VERSION 3.0)

# project information
project(semiprof)
enable_language(CXX)

# set path for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# turn on profiler
#add_definitions(-DSEMIPROF)

# create the semiprof library
add_library(semiprof src/profiler.cpp)

# compile the examples
add_subdirectory(examples)
