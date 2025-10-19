add_custom_target(tests COMMENT "Builds all tests.")

# CACHE GTest main into an OBJECT library to avoid recompiling.
#
add_library(main_gtest_mpi OBJECT main_gtest_mpi.cpp)
target_link_libraries(main_gtest_mpi PRIVATE MPI::MPI_CXX gtest_mpi)

add_library(main_gtest OBJECT main_gtest.cpp)
target_link_libraries(main_gtest PRIVATE gtest)

#  Unit tests
#
function(add_cosma_mpi_test test_name num_procs)
    set(tgt_ test.${test_name})
    add_executable(${tgt_} ${test_name}.cpp)
    target_link_libraries(${tgt_} PRIVATE main_gtest_mpi gtest_mpi ${ARGN})
    target_include_directories(${tgt_} PRIVATE .)
    add_test(NAME ${tgt_}
             WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
             COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${num_procs}
                     ${MPIEXEC_PREFLAGS} ./${tgt_} ${MPIEXEC_POSTFLAGS})
    add_dependencies(tests ${tgt_})
    install(TARGETS ${tgt_} DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()

add_executable(test.mapper mapper.cpp)
target_link_libraries(test.mapper PRIVATE main_gtest gtest cosma)
add_test(NAME test.mapper COMMAND test.mapper)
add_dependencies(tests test.mapper)
install(TARGETS test.mapper DESTINATION "${CMAKE_INSTALL_BINDIR}")

# BFloat16 basic test (non-MPI)
add_executable(test.bfloat16_basic test_bfloat16_basic.cpp)
target_link_libraries(test.bfloat16_basic PRIVATE cosma)
add_test(NAME test.bfloat16_basic COMMAND test.bfloat16_basic)
add_dependencies(tests test.bfloat16_basic)
install(TARGETS test.bfloat16_basic DESTINATION "${CMAKE_INSTALL_BINDIR}")

# BFloat16 MPI test (2 ranks)
add_executable(test.bfloat16_mpi test_bfloat16_mpi.cpp)
target_link_libraries(test.bfloat16_mpi PRIVATE cosma MPI::MPI_CXX)
add_test(NAME test.bfloat16_mpi
         WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2
                 ${MPIEXEC_PREFLAGS} ./test.bfloat16_mpi ${MPIEXEC_POSTFLAGS})
add_dependencies(tests test.bfloat16_mpi)
install(TARGETS test.bfloat16_mpi DESTINATION "${CMAKE_INSTALL_BINDIR}")

# BFloat16 debug test for custom strategy
add_executable(debug.bf16_custom debug_bf16_custom_strategy.cpp)
target_link_libraries(debug.bf16_custom PRIVATE cosma MPI::MPI_CXX)
add_dependencies(tests debug.bf16_custom)

# BFloat16 MPI byte order test
add_executable(test.bf16_bytes test_bf16_mpi_byte_order.cpp)
target_link_libraries(test.bf16_bytes PRIVATE cosma MPI::MPI_CXX)
add_dependencies(tests test.bf16_bytes)

# Buffer size debug test
add_executable(debug.buffer_sizes debug_buffer_sizes.cpp)
target_link_libraries(debug.buffer_sizes PRIVATE cosma MPI::MPI_CXX)
add_dependencies(tests debug.buffer_sizes)

# Minimal reproducer for BF16 custom strategy bug
add_executable(minimal_bf16_bug minimal_bf16_custom_bug.cpp)
target_link_libraries(minimal_bf16_bug PRIVATE cosma MPI::MPI_CXX)
add_dependencies(tests minimal_bf16_bug)
install(TARGETS minimal_bf16_bug DESTINATION "${CMAKE_INSTALL_BINDIR}")

# Debug tool for BF16 buffer sizes during K-split
add_executable(debug_bf16_buffer_sizes debug_bf16_buffer_sizes.cpp)
target_link_libraries(debug_bf16_buffer_sizes PRIVATE cosma MPI::MPI_CXX)
add_dependencies(tests debug_bf16_buffer_sizes)
install(TARGETS debug_bf16_buffer_sizes DESTINATION "${CMAKE_INSTALL_BINDIR}")

# BFloat16 backend benchmark (MKL vs OpenBLAS)
add_executable(benchmark.bf16_backends benchmark_bf16_backends.cpp)
target_link_libraries(benchmark.bf16_backends PRIVATE cosma)
add_dependencies(tests benchmark.bf16_backends)
install(TARGETS benchmark.bf16_backends DESTINATION "${CMAKE_INSTALL_BINDIR}")

# BFloat16 distributed multiply test (8 ranks)
add_cosma_mpi_test(bfloat16_multiply 8 cosma)

if(NOT COSMA_SCALAPACK MATCHES "OFF")
    add_cosma_mpi_test(pdgemm 16 cosma_pxgemm_cpp)
endif()

add_cosma_mpi_test(multiply_using_layout 4 cosma)
add_cosma_mpi_test(multiply 16 cosma)
add_cosma_mpi_test(scalar_matmul 8 cosma)
