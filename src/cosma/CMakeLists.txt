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

if(COSMA_WITH_INSTALL)
    install(TARGETS cosma
            EXPORT cosma_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

if(ScaLAPACK_TARGET)
    add_library(cosma_pxgemm STATIC scalapack.cpp
                                    pgemm.cpp
                                    pxgemm.cpp
    )
    target_link_libraries(cosma_pxgemm PUBLIC cosma 
                                              ${ScaLAPACK_TARGET}
    )

    if (COSMA_WITH_FORTRAN_BINDINGS)
        add_library(cosma_pxgemm_fortran STATIC pxgemm.f90)
        target_link_libraries(cosma_pxgemm_fortran PUBLIC cosma_pxgemm)
    endif()

    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma_pxgemm PRIVATE semiprof)
        target_compile_definitions(cosma_pxgemm PRIVATE COSMA_WITH_PROFILING)
        if (COSMA_WITH_FORTRAN_BINDINGS)
            target_link_libraries(cosma_pxgemm_fortran PRIVATE semiprof)
            target_compile_definitions(cosma_pxgemm_fortran PRIVATE COSMA_WITH_PROFILING)
        endif()
    endif()


    if(COSMA_WITH_INSTALL)
        install(TARGETS cosma_pxgemm
                EXPORT cosma_targets
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
        if (COSMA_WITH_FORTRAN_BINDINGS)
            install(TARGETS cosma_pxgemm_fortran
                    EXPORT cosma_targets
                    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
        endif()
    endif()
endif()
