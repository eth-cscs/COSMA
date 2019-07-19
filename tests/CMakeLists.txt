option(COSMA_IS_OPENMPI "Is MPI vendor OpenMPI?" OFF)
if (COSMA_IS_OPENMPI)
    set(OVERSUBSCRIBE "--oversubscribe")
endif()

add_custom_target(tests COMMENT "Builds all tests if any.")


################
#  Build test  #
################
add_executable(test.cosma cosma_test.cpp)
target_link_libraries(test.cosma PRIVATE cosma)
target_include_directories(test.cosma PRIVATE .)

if(ScaLAPACK_FOUND)
    add_executable(test.cosma_pdgemm cosma_pdgemm_test.cpp)
    target_link_libraries(test.cosma_pdgemm PRIVATE cosma options grid2grid MKL::ScaLAPACK)
    target_include_directories(test.cosma_pdgemm PRIVATE .)
endif()

add_executable(test.strategy test_strategy.cpp)
target_link_libraries(test.strategy PRIVATE cosma)
add_test(NAME test.strategy COMMAND test.strategy)
add_dependencies(tests test.strategy)

################
#  unit gtests #
################
add_executable(test.mapper test_mapper.cpp unit_test.cpp)
target_link_libraries(test.mapper PRIVATE gtest cosma)
target_include_directories(test.mapper PRIVATE .)
add_test(NAME test.mapper COMMAND test.mapper)
add_dependencies(tests test.mapper)

add_executable(test.multiply test_multiply.cpp unit_test_mpi.cpp)
target_link_libraries(test.multiply PRIVATE gtest_mpi cosma)
target_include_directories(test.multiply PRIVATE .)
add_test(NAME test.multiply
         COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
                 ${MPIEXEC_NUMPROC_FLAG} 16 test.multiply)
add_dependencies(tests test.multiply)

if(ScaLAPACK_FOUND)
    add_executable(test.pdgemm test_pdgemm.cpp unit_test_mpi.cpp)
    target_link_libraries(test.pdgemm PRIVATE gtest_mpi cosma MKL::ScaLAPACK grid2grid options)
    target_include_directories(test.pdgemm PRIVATE .)
    add_test(NAME test.pdgemm
             COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
                     ${MPIEXEC_NUMPROC_FLAG} 6 test.pdgemm)
    add_dependencies(tests test.pdgemm)
endif()

add_executable(test.scalar_matmul scalar_matmul.test.cpp unit_test_mpi.cpp)
target_link_libraries(test.scalar_matmul PRIVATE gtest_mpi cosma)
target_include_directories(test.scalar_matmul PRIVATE .)
add_test(NAME test.scalar_matmul
         COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
                 ${MPIEXEC_NUMPROC_FLAG} 8 test.scalar_matmul)
add_dependencies(tests test.scalar_matmul)
