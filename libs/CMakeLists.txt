add_subdirectory(options)
add_subdirectory(semiprof)
# add_subdirectory(semiprof)
add_subdirectory(grid2grid)

if(COSMA_WITH_CUDA)
    add_subdirectory(Tiled-MM)
endif()
