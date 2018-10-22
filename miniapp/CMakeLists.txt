cmake_minimum_required(VERSION 3.5)

################
#  Build test  #
################
set(executables "cosma-miniapp" "test-strategy" "cosma-statistics")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${cosma_deps})
endforeach()

add_executable("cuda_test" "cuda_test.cpp")
target_link_libraries("cuda_test" ${cuda_deps})
