set(INSTALLED_TARGETS_LIST "")
if(NOT TARGET costa)
    # original grid2grid files
    set(costa_src_files 
                      grid2grid/grid_layout.hpp
                      grid2grid/block.cpp
                      grid2grid/grid2D.cpp
                      grid2grid/interval.cpp
                      grid2grid/scalapack_layout.cpp
                      grid2grid/communication_data.cpp
                      grid2grid/grid_cover.cpp
                      grid2grid/ranks_reordering.cpp
                      grid2grid/transform.cpp
                      grid2grid/transformer.hpp
                      layout.cpp
    )
    set(costa_prefixed_scalapack_src_files 
                        scalapack.cpp
                        # templatized cpp implementations
                        pxtran/costa_pxtran.cpp
                        pxgemr2d/costa_pxgemr2d.cpp
                        # scalapck api with costa prefix
                        pxtran/prefixed_pxtran.cpp
                        pxgemr2d/prefixed_pxgemr2d.cpp
    )
    set(costa_scalapack_src_files 
                        scalapack.cpp
                        # templatized cpp implementations
                        pxtran/costa_pxtran.cpp
                        pxgemr2d/costa_pxgemr2d.cpp
                        # scalapck api overwritten
                        pxtran/pxtran.cpp
                        pxgemr2d/pxgemr2d.cpp
    )

    add_library(costa ${costa_src_files})
    target_include_directories(costa PUBLIC 
        $<BUILD_INTERFACE:${costa_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${costa_SOURCE_DIR}/src/grid2grid>
        $<BUILD_INTERFACE:${costa_SOURCE_DIR}/src/pxgemr2d>
        $<BUILD_INTERFACE:${costa_SOURCE_DIR}/src/pxtran>
    )
    target_compile_features(costa PUBLIC cxx_std_14)
    target_link_libraries(costa PUBLIC MPI::MPI_CXX
                                       OpenMP::OpenMP_CXX)

    if(COSTA_WITH_PROFILING)
        target_link_libraries(costa PRIVATE
            $<BUILD_INTERFACE:semiprof>
        )
        target_compile_definitions(costa PRIVATE
            $<BUILD_INTERFACE:COSTA_WITH_PROFILING>
        )
    endif()

    list(APPEND INSTALLED_TARGETS_LIST "costa")
endif()

if(NOT TARGET costa_scalapack)
    add_library(costa_scalapack ${costa_scalapack_src_files})

    target_link_libraries(costa_scalapack PUBLIC costa 
        ${SCALAPACK_TARGET}
    )
    if(COSTA_WITH_PROFILING)
        target_link_libraries(costa_scalapack PRIVATE
            $<BUILD_INTERFACE:semiprof>
        )
        target_compile_definitions(costa_scalapack PRIVATE
            $<BUILD_INTERFACE:COSTA_WITH_PROFILING>
        )
    endif()
    list(APPEND INSTALLED_TARGETS_LIST "costa_scalapack")
endif()

if(NOT TARGET costa_prefixed_scalapack)
    add_library(costa_prefixed_scalapack ${costa_prefixed_scalapack_src_files})

    target_link_libraries(costa_prefixed_scalapack PUBLIC costa 
        ${SCALAPACK_TARGET}
    )
    if(COSTA_WITH_PROFILING)
        target_link_libraries(costa_prefixed_scalapack PRIVATE
            $<BUILD_INTERFACE:semiprof>
        )
        target_compile_definitions(costa_prefixed_scalapack PRIVATE
            $<BUILD_INTERFACE:COSTA_WITH_PROFILING>
        )
    endif()
    list(APPEND INSTALLED_TARGETS_LIST "costa_prefixed_scalapack")
endif()

if(COSTA_WITH_INSTALL AND INSTALLED_TARGETS_LIST)
    install(TARGETS ${INSTALLED_TARGETS_LIST}
            EXPORT costa_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(EXPORT costa_targets
            FILE costaTargets.cmake
            NAMESPACE costa::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/costa")
endif()

