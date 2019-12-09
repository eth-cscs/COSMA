################
#  Build test  #
################
set(executables "cosma-miniapp" "blocking_vs_non_blocking" "cosma-statistics" "scalars_miniapp")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE cosma options)
endforeach()

if(COSMA_SCALAPACK)
    add_executable(pdgemm-miniapp "pdgemm-miniapp.cpp")
    target_link_libraries(pdgemm-miniapp PRIVATE cosma_pxgemm options)
    if (COSMA_WITH_PROFILING)
        target_link_libraries(pdgemm-miniapp PRIVATE semiprof)
        target_compile_definitions(pdgemm-miniapp PRIVATE COSMA_WITH_PROFILING)
    endif()
endif()
