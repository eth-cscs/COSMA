################
#  Build test  #
################
add_executable(cosma-test "cosma-test.cpp")
target_link_libraries(cosma-test cosma ${cosma_deps})

################
#  unit gtests #
################
set(TEST_SOURCES test_mapper.cpp unit_test.cpp)
add_executable(unit_test.exe ${TEST_SOURCES})
target_link_libraries(unit_test.exe gtest cosma ${cosma_deps})
# target_include_directories(unit_test.exe PUBLIC ${PROJECT_SOURCE_DIR}/libs/gtest)

set(TEST_SOURCES_MPI test_multiply.cpp unit_test_mpi.cpp)
add_executable(unit_test_mpi.exe ${TEST_SOURCES_MPI})
target_link_libraries(unit_test_mpi.exe gtest_mpi cosma ${cosma_deps})
# target_include_directories(unit_test_mpi.exe PUBLIC ${PROJECT_SOURCE_DIR}/libs/gtest ${PROJECT_SOURCE_DIR}/libs/gtest_mpi)

add_test(no_mpi_tests unit_test.exe)

if (MPI_FOUND)

    option(REMOTE "--oversubscribe flag for OpenMPI" OFF)
    set(OVERSUBSCRIBE "--oversubscribe")
    if(REMOTE)
        set(OVERSUBSCRIBE "")
    endif()

    add_test(NAME with_mpi_tests COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 unit_test_mpi.exe)
endif ()









