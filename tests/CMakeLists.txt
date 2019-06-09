option(IS_OPENMPI "Is MPI vendor OpenMPI?" OFF)
if (IS_OPENMPI)
    set(OVERSUBSCRIBE "--oversubscribe")
endif()

################
#  Build test  #
################
add_executable(test.cosma cosma-test.cpp)
target_link_libraries(test.cosma PRIVATE cosma)
target_include_directories(test.cosma PRIVATE .)
#add_test(NAME test.cosma COMMAND test.cosma)

################
#  unit gtests #
################
add_executable(test.strategy test-strategy.cpp)
target_link_libraries(test.strategy PRIVATE cosma)
add_test(NAME test.strategy COMMAND test.strategy)

add_executable(test.mapper test_mapper.cpp unit_test.cpp)
target_link_libraries(test.mapper PRIVATE gtest cosma)
target_include_directories(test.mapper PRIVATE .)
add_test(NAME test.mapper COMMAND test.mapper)

add_executable(test.multiply test_multiply.cpp unit_test_mpi.cpp)
target_link_libraries(test.multiply PRIVATE gtest_mpi cosma)
target_include_directories(test.multiply PRIVATE .)
add_test(NAME test.multiply
         COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
                 ${MPIEXEC_NUMPROC_FLAG} 16 test.multiply)
