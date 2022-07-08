cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(gtest_mpi CXX)

add_subdirectory(external/gtest)
add_library(gtest_mpi INTERFACE)
target_include_directories(gtest_mpi INTERFACE ${gtest_mpi_SOURCE_DIR}/include)
target_link_libraries(gtest_mpi INTERFACE gtest)
target_compile_features(gtest_mpi INTERFACE cxx_std_11)
