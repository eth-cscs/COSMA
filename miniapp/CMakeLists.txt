################
#  Build test  #
################
set(executables "layout_miniapp" "cosma_miniapp" "cosma_statistics")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE cosma cxxopts::cxxopts)
    install(TARGETS ${exec} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()

if(NOT COSMA_SCALAPACK MATCHES "OFF")
    set(scalapack_executables "pxgemm_miniapp")
    foreach(exec ${scalapack_executables})
        add_executable(${exec} "${exec}.cpp")
        target_link_libraries(${exec} PRIVATE cosma_pxgemm_cpp cxxopts::cxxopts)
        install(TARGETS ${exec} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
        if (COSMA_WITH_PROFILING)
            target_link_libraries(${exec} PRIVATE semiprof::semiprof)
            target_compile_definitions(${exec} PRIVATE COSMA_WITH_PROFILING)
        endif()
    endforeach()
endif()
