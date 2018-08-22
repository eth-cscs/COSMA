cmake_minimum_required(VERSION 3.5)

project(ubench)

include_directories("${PRJ_ROOT_DIR}/src")
add_lib_executable(ubench-allgather cosma ${mpi_deps})

