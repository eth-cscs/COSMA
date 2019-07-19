################
#  Build test  #
################
set(executables "cosma-miniapp" "blocking_vs_non_blocking" "cosma-statistics" "scalars_miniapp")

if (ScaLAPACK_FOUND)
    list(APPEND executables "pdgemm-miniapp")
endif()

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE cosma options MKL::ScaLAPACK)
endforeach()

#add_executable(cosma-temp "cosma-temp.cpp")
#target_link_libraries(cosma-temp cosma ${cosma_deps} /users/kabicm/local/mpip/lib/libmpiP.so -lm -liberty)

