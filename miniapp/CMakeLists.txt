################
#  Build test  #
################
set(executables "cosma-miniapp" "blocking_vs_non_blocking")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${cosma_deps})
endforeach()

add_executable(cosma-statistics "cosma-statistics.cpp")
target_link_libraries(cosma-statistics cosma)

#add_executable(cosma-temp "cosma-temp.cpp")
#target_link_libraries(cosma-temp cosma ${cosma_deps} /users/kabicm/local/mpip/lib/libmpiP.so -lm -liberty)

