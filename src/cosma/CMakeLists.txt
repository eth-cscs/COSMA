set(INSTALLED_TARGETS_LIST "")
if(NOT TARGET cosma)
    set(cosma_src_files blas.cpp
                        buffer.cpp
                        communicator.cpp
                        context.cpp
                        interval.cpp
                        layout.cpp
                        local_multiply.cpp
                        mapper.cpp
                        math_utils.cpp
                        matrix.cpp
                        memory_pool.cpp
                        mpi_attribute.cpp
                        multiply.cpp
                        one_sided_communicator.cpp
                        strategy.cpp
                        two_sided_communicator.cpp
    )
    add_library(cosma STATIC ${cosma_src_files})
    target_include_directories(cosma PUBLIC $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>)
    target_compile_features(cosma PUBLIC cxx_std_14)
    target_link_libraries(cosma PUBLIC MPI::MPI_CXX
                                       grid2grid
                                       ${BLAS_TARGET}
                                PRIVATE options
    )
    target_compile_definitions(cosma PRIVATE ${BLAS_DEF}
    )
    
    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma PRIVATE semiprof)
        target_compile_definitions(cosma PRIVATE COSMA_WITH_PROFILING)
    endif()

    list(APPEND INSTALLED_TARGETS_LIST "cosma")
endif()

if(ScaLAPACK_TARGET AND NOT TARGET cosma_pxgemm)
    add_library(cosma_pxgemm STATIC scalapack.cpp
                                    pgemm.cpp
                                    pxgemm.cpp
    )
    target_link_libraries(cosma_pxgemm PUBLIC cosma 
                                              ${ScaLAPACK_TARGET}
    )
    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma_pxgemm PRIVATE semiprof)
        target_compile_definitions(cosma_pxgemm PRIVATE COSMA_WITH_PROFILING)
    endif()
    list(APPEND INSTALLED_TARGETS_LIST "cosma_pxgemm")

    if (COSMA_WITH_FORTRAN_BINDINGS)
        add_library(cosma_pxgemm_fortran STATIC pxgemm.f90)
        target_link_libraries(cosma_pxgemm_fortran PUBLIC cosma_pxgemm)
        list(APPEND INSTALLED_TARGETS_LIST "cosma_pxgemm_fortran")
        INSTALL ( CODE
            "EXECUTE_PROCESS (COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"${PROJECT_BINARY_DIR}/src/cosma\" \"${CMAKE_INSTALL_PREFIX}/include/cosma/\")"
            )
    endif()
endif()

if(COSMA_WITH_INSTALL AND INSTALLED_TARGETS_LIST)
    install(TARGETS ${INSTALLED_TARGETS_LIST}
            EXPORT cosma_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(EXPORT cosma_targets
            FILE cosmaTargets.cmake
            NAMESPACE cosma::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cosma")
endif()
