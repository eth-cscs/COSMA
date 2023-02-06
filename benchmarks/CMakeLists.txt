include(find_cuda_version)

################
#  Build test  #
################
set(executables "ubench-allgather"
                "allgather-volume"
                "sendrecv"
                "reduce-scatter"
                "blocking_vs_non_blocking"
                "dgemm_perf_model")

# if (${COSMA_BLAS} STREQUAL "MKL")
#     list(APPEND executables "transpose")
# endif()

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} cosma ${BLAS_TARGET})
endforeach()

if (${COSMA_BLAS} STREQUAL "CUDA")
    find_cuda_version()
    # check if cuda toolkit version >= 10.1
    # which is needed for cublasLt (used in the benchmark)
    if (CUDA_TOOLKIT_MAJOR_VERSION GREATER 10 OR
            (CUDA_TOOLKIT_MAJOR_VERSION EQUAL 10 AND CUDA_TOOLKIT_MINOR_VERSION GREATER_EQUAL 1))
        add_executable(gpu_gemm_cublas "gpu_gemm_cublas.cpp")
        target_link_libraries(gpu_gemm_cublas cosma ${BLAS_TARGET} cublasLt cublas)
        target_compile_definitions(gpu_gemm_cublas PRIVATE ${BLAS_DEF})
    endif()
endif()
