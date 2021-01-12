################
#  Build test  #
################
# COSTA-only executables
set(executables "example0" "example1")
foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE costa cxxopts)
    target_include_directories(${exec} PUBLIC "${cosma_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()

# COSTA without scalapack executable
set(scalapack_executables "")

if (SCALAPACK_TARGET) 
    foreach(exec ${scalapack_executables})
        add_executable(${exec} "${exec}.cpp")
        target_link_libraries(${exec} PRIVATE costa_scalapack cxxopts)
        target_include_directories(${exec} PUBLIC "${cosma_SOURCE_DIR}/libs/cxxopts/include/")
        if (COSTA_WITH_PROFILING)
            target_link_libraries(${exec} PRIVATE semiprof)
            target_compile_definitions(${exec} PRIVATE COSTA_WITH_PROFILING)
        endif()
    endforeach()
endif()
