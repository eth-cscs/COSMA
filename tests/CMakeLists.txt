################
#  Build test  #
################
add_executable(cosma-test cosma-test.cpp)
target_link_libraries(cosma-test PRIVATE cosma)
target_include_directories(cosma-test PRIVATE .)

################
#  unit gtests #
################
add_executable(unit_test.exe test_mapper.cpp unit_test.cpp)
target_link_libraries(unit_test.exe PRIVATE gtest cosma)
target_include_directories(unit_test.exe PRIVATE .)
# target_include_directories(unit_test.exe PUBLIC ${PROJECT_SOURCE_DIR}/libs/gtest)

add_executable(unit_test_mpi.exe test_multiply.cpp unit_test_mpi.cpp)
target_link_libraries(unit_test_mpi.exe PRIVATE gtest_mpi cosma)
target_include_directories(unit_test_mpi.exe PRIVATE .)
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

