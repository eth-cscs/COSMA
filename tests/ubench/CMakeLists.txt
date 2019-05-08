################
#  Build test  #
################
set(executables "ubench-allgather"
                "allgather-volume"
                "sendrecv"
                "reduce-scatter"
                "dgemm_perf_model"
                "test-strategy")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${cosma_deps})
endforeach()

# if(COSMA_SCALAPACK_LIBRARY)
#     add_executable(scalapack_transformer "scalapack_transformer.cpp")
#     target_link_libraries(scalapack_transformer cosma ${cosma_deps} ${scalapack_deps})
# endif()

# link to mpiP for profiling
#add_executable(sendrecv "sendrecv.cpp")
#target_link_libraries(sendrecv cosma ${cosma_deps} /users/kabicm/local/mpip/lib/libmpiP.a -lm -liberty)


# add_executable(gpu_gemm_libsci_acc "gpu_gemm_libsci_acc.cpp")
# target_link_libraries(gpu_gemm_libsci_acc)

#add_executable(gpu_gemm_cublas "gpu_gemm_cublas.cpp")
#target_link_libraries(gpu_gemm_cublas cosma ${cosma_deps})
