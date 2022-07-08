cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(gtest CXX)

add_library(gtest STATIC src/gtest-all.cpp)
target_include_directories(gtest PUBLIC ${gtest_SOURCE_DIR}/include)
target_compile_features(gtest PUBLIC cxx_std_11)
