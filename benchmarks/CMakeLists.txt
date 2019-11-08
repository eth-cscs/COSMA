################
#  Build test  #
################
set(executables "ubench-allgather"
                "allgather-volume"
                "sendrecv"
                "reduce-scatter"
                "dgemm_perf_model")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${cosma_deps})
endforeach()

if (${COSMA_BLAS} STREQUAL "MKL")
    add_executable(transpose "transpose.cpp")
    target_link_libraries(transpose cosma ${cosma_deps})
endif()

