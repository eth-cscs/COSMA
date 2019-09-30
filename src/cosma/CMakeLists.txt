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
                                    $<$<BOOL:${COSMA_WITH_PROFILING}>:semiprof>
)
target_compile_definitions(cosma PRIVATE ${BLAS_DEF}
                                         $<$<BOOL:${COSMA_WITH_PROFILING}>:COSMA_WITH_PROFILING>
)

if(COSMA_WITH_INSTALL)
    install(TARGETS cosma
            EXPORT cosma_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

if(ScaLAPACK_TARGET)
    add_library(cosma_with_scalapack STATIC scalapack.cpp
                                            pgemm.cpp
    )
    target_link_libraries(cosma_with_scalapack PUBLIC cosma 
                                                      ${ScaLAPACK_TARGET}
                                               PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:semiprof>
    )
    target_compile_definitions(cosma_with_scalapack PRIVATE $<$<BOOL:${COSMA_WITH_PROFILING}>:COSMA_WITH_PROFILING>)

    if(COSMA_WITH_INSTALL)
        install(TARGETS cosma_with_scalapack
                EXPORT cosma_targets
                LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    endif()
endif()
