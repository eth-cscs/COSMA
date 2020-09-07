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
                        multiply.cpp
                        one_sided_communicator.cpp
                        strategy.cpp
                        two_sided_communicator.cpp
                        random_generator.hpp
                        cinterface.cpp
                        environment_variables.cpp
    )

    if (GPU)
        list(APPEND cosma_src_files "pinned_buffers.cpp")
    endif()

    add_library(cosma ${cosma_src_files})
    target_include_directories(cosma PUBLIC $<BUILD_INTERFACE:${cosma_SOURCE_DIR}/src>)
    target_compile_features(cosma PUBLIC cxx_std_14)
    target_link_libraries(cosma PUBLIC MPI::MPI_CXX
                                       grid2grid
                                PRIVATE ${BLAS_TARGET}
    )
    target_compile_definitions(cosma PRIVATE ${BLAS_DEF}
    )

    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma PRIVATE semiprof)
        target_compile_definitions(cosma PRIVATE COSMA_WITH_PROFILING)
    endif()

    list(APPEND INSTALLED_TARGETS_LIST "cosma")
endif()

# if ScaLAPACK is found and cosma_pxgemm library is not already created
# then create it here and link it to the profiler if needed
if(ScaLAPACK_TARGET AND NOT TARGET cosma_pxgemm)
    add_library(cosma_pxgemm scalapack.cpp
                             pxgemm_params.hpp
                             cosma_pxgemm.cpp
                             pxgemm.cpp
    )
    target_link_libraries(cosma_pxgemm PUBLIC cosma 
                                              ${BLAS_TARET}
                                              ${ScaLAPACK_TARGET}
    )
    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma_pxgemm PRIVATE semiprof)
        target_compile_definitions(cosma_pxgemm PRIVATE COSMA_WITH_PROFILING)
    endif()
    list(APPEND INSTALLED_TARGETS_LIST "cosma_pxgemm")
endif()

# this is a library exposing the prefixed scalapack API (with cosma/COSMA prefix)
# it is aimed for users who don't want to overwrite the available scalapack API with cosma.
# if ScaLAPACK is found and cosma_prefixed_pxgemm library is not already created
# then create it here and link it to the profiler if needed
if(ScaLAPACK_TARGET AND NOT TARGET cosma_prefixed_pxgemm)
    add_library(cosma_prefixed_pxgemm scalapack.cpp
                             pxgemm_params.hpp
                             prefixed_pxgemm.cpp
                             cosma_pxgemm.cpp
    )
    target_link_libraries(cosma_prefixed_pxgemm PUBLIC cosma 
                                              ${BLAS_TARET}
                                              ${ScaLAPACK_TARGET}
    )
    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma_prefixed_pxgemm PRIVATE semiprof)
        target_compile_definitions(cosma_prefixed_pxgemm PRIVATE COSMA_WITH_PROFILING)
    endif()
    list(APPEND INSTALLED_TARGETS_LIST "cosma_prefixed_pxgemm")
endif()

# the following library is aimed only for testing purposes
# it provides templated cosma::pxgemm call without
# pxgemm.h, so that pxgemm calls of scalapack are not overwritten
# and can still be compared to scalapack for correctness check
if(ScaLAPACK_TARGET AND NOT TARGET cosma_pxgemm_cpp)
    add_library(cosma_pxgemm_cpp scalapack.cpp
                                 pxgemm_params.hpp
                                 cosma_pxgemm.cpp
    )
    target_link_libraries(cosma_pxgemm_cpp PUBLIC cosma 
                                              ${BLAS_TARET}
                                              ${ScaLAPACK_TARGET}
    )
    if(COSMA_WITH_PROFILING)
        target_link_libraries(cosma_pxgemm_cpp PRIVATE semiprof)
        target_compile_definitions(cosma_pxgemm_cpp PRIVATE COSMA_WITH_PROFILING)
    endif()
    list(APPEND INSTALLED_TARGETS_LIST "cosma_pxgemm_cpp")
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
