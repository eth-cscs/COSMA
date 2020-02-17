################
#  Build test  #
################
set(executables "ubench-allgather"
                "allgather-volume"
                "sendrecv"
                "reduce-scatter"
                "blocking_vs_non_blocking"
                "dgemm_perf_model")

if (${COSMA_BLAS} STREQUAL "MKL")
    list(APPEND executables "transpose")
endif()

if (${COSMA_BLAS} STREQUAL "CUDA")
    list(APPEND executables "gpu_gemm_cublas")
endif()

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${cosma_deps})
endforeach()

if (${COSMA_BLAS} STREQUAL "CUDA")
    target_link_libraries(gpu_gemm_cublas ${BLAS_TARGET} cublasLt cublas)
    target_compile_definitions(gpu_gemm_cublas PRIVATE ${BLAS_DEF})
endif()
