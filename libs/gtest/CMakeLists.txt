PROJECT(gtest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gtest gtest-all.cpp)
target_include_directories(gtest PUBLIC ${PROJECT_SOURCE_DIR})

add_library(gtest_mpi gtest-all.cpp)
target_include_directories(gtest_mpi PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/gtest_mpi)
