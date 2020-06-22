################
#  Build test  #
################
set(executables "cosma_miniapp" "cosma_statistics")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE cosma cxxopts)
    target_include_directories(${exec} PUBLIC "${cosma_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()

if(COSMA_SCALAPACK)
    set(scalapack_executables "pxgemm_miniapp")
    foreach(exec ${scalapack_executables})
        add_executable(${exec} "${exec}.cpp")
        target_link_libraries(${exec} PRIVATE cosma_pxgemm_cpp)
        target_include_directories(${exec} PUBLIC "${cosma_SOURCE_DIR}/libs/cxxopts/include/")
        if (COSMA_WITH_PROFILING)
            target_link_libraries(${exec} PRIVATE semiprof)
            target_compile_definitions(${exec} PRIVATE COSMA_WITH_PROFILING)
        endif()
    endforeach()
endif()
