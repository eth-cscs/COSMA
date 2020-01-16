################
#  Build test  #
################
set(executables "cosma-miniapp" "blocking_vs_non_blocking" "cosma-statistics" "scalars_miniapp")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE cosma options)
endforeach()

if(COSMA_SCALAPACK)
    set(scalapack_executables "pdgemm-miniapp" "pdgemm-detailed-miniapp")
    foreach(exec ${scalapack_executables})
        add_executable(${exec} "${exec}.cpp")
        target_link_libraries(${exec} PRIVATE cosma_pxgemm options)
        if (COSMA_WITH_PROFILING)
            target_link_libraries(${exec} PRIVATE semiprof)
            target_compile_definitions(${exec} PRIVATE COSMA_WITH_PROFILING)
        endif()
    endforeach()
endif()
