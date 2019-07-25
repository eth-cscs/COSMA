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
                    multiply.cpp
                    one_sided_communicator.cpp
                    strategy.cpp
                    two_sided_communicator.cpp
)
add_library(cosma STATIC ${cosma_src_files})

target_include_directories(cosma PUBLIC 
    $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>
)
target_compile_features(cosma PUBLIC cxx_std_14)

target_link_libraries(cosma PUBLIC MPI::MPI_CXX
                                   grid2grid
                            PRIVATE options
                                    semiprof)
if(COSMA_WITH_PROFILING)
    target_compile_definitions(cosma PRIVATE SEMIPROF)
endif()

if(COSMA_WITH_MKL)
    target_link_libraries(cosma PRIVATE MKL::MKL)
    target_compile_definitions(cosma PRIVATE COSMA_WITH_MKL)
endif()

if(COSMA_WITH_TILEDMM)
    # These need to be public because of context hpp|cpp
    #
    target_link_libraries(cosma PUBLIC Tiled-MM)
    target_compile_definitions(cosma PUBLIC COSMA_HAVE_GPU)
endif()

if(ScaLAPACK_FOUND)
    add_library(cosma_with_scalapack STATIC scalapack.cpp
                                            pgemm.cpp
    )
    target_link_libraries(cosma_with_scalapack PUBLIC cosma MKL::ScaLAPACK)
endif()

if(COSMA_WITH_INSTALL)
    set(COSMA_TGT "cosma")
    if(ScaLAPACK_FOUND)
        list(APPEND COSMA_TGT "cosma_with_scalapack")
    endif()
    install(TARGETS ${COSMA_TGT}
            EXPORT cosma_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
