################
#  Build test  #
################
set(executables "cosma-miniapp" "blocking_vs_non_blocking" "cosma-statistics" "scalars_miniapp")

if(COSMA_WITH_MKL_ScaLAPACK OR COSMA_WITH_NETLIB_ScaLAPACK)
    add_executable(pdgemm-miniapp "pdgemm-miniapp.cpp")
    target_link_libraries(pdgemm-miniapp PRIVATE cosma_pxgemm options)
    if (COSMA_WITH_PROFILING)
        target_link_libraries(pdgemm-miniapp PRIVATE semiprof)
        target_compile_definitions(pdgemm-miniapp PRIVATE COSMA_WITH_PROFILING)
    endif()
endif()

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE cosma options)
endforeach()

#add_executable(cosma-temp "cosma-temp.cpp")
#target_link_libraries(cosma-temp cosma ${cosma_deps} /users/kabicm/local/mpip/lib/libmpiP.so -lm -liberty)

