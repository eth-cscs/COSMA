cmake_minimum_required(VERSION 3.5)

################
#  Build test  #
################
set(executables "cosma-miniapp" "temp-miniapp" "test-strategy" "cosma-statistics")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${cosma_deps})
endforeach()
