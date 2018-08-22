cmake_minimum_required(VERSION 3.5)

project(tests)

set(BIN_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TEST_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

####################
#  Lib Dependancy  #
####################

include_directories("${PRJ_ROOT_DIR}/src")

################
#  Build test  #
################

if (COSMA_WITH_CUDA)
    cuda_add_lib_executable(cosma-test
        cosma
        cosmacu
        ${cosma_deps})
else()
    add_lib_executable(cosma-test
        cosma
        ${cosma_deps})
endif()

################
#  unit gtests #
################
set(TEST_SOURCES test_mapper.cpp unit_test.cpp)

set(target unit_test.exe)
add_executable(unit_test.exe ${TEST_SOURCES})
target_link_libraries(${target} gtest)
target_link_libraries(${target} cosma)
add_test(layout-and-mapper-tests unit_test.exe)

# debug with
# mpirun -np 1 xterm -e lldb -o run -- ./cosma-test  ...

if (MPI_FOUND)
# test the correctness of cosma
    add_test(NAME cosma-test-buffers-test COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 4 -n 4 -k 4 -P 4 -s dm2,bn2,bn2)

    add_test(NAME cosma-test-bbdd-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 30 -n 35 -k 40 -P 4 -s bm2,bn2,dk2,dm2)
    add_test(NAME cosma-test-bbdd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 30 -n 35 -k 40 -P 4)

    add_test(NAME cosma-test-bdb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 8 -n 4 -k 2 -P 4 -s bm2,dm2,bn2)
    add_test(NAME cosma-test-bdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 8 -n 4 -k 2 -P 4)

    add_test(NAME cosma-test-ddb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 2 cosma-test -m 8 -n 8 -k 2 -P 2 -s dm2,dm2,bn2)
    add_test(NAME cosma-test-ddb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 2 cosma-test -m 8 -n 8 -k 2 -P 2)

    add_test(NAME cosma-test-bb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 16 -n 4 -k 4 -P 4 -s bm2,bm2)
    add_test(NAME cosma-test-bb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 16 -n 4 -k 4 -P 4)

    add_test(NAME cosma-test-db-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 3 cosma-test -m 20 -n 20 -k 20 -P 3 -s dk2,bm3)
    add_test(NAME cosma-test-db COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 3 cosma-test -m 20 -n 20 -k 20 -P 3)

    add_test(NAME cosma-test-bbbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 cosma-test -m 16 -n 16 -k 16 -P 16 -s bm2,bn2,bk2,bm2)
    add_test(NAME cosma-test-bbbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 cosma-test -m 16 -n 16 -k 16 -P 16)

    add_test(NAME cosma-test-dddd-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 1 cosma-test -m 16 -n 16 -k 16 -P 1 -s dm2,dn2,dm2,dn2)
    add_test(NAME cosma-test-dddd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 1 cosma-test -m 16 -n 16 -k 16 -P 1)

    add_test(NAME cosma-test-ddbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 20 -n 30 -k 25 -P 4 -s dm2,dn2,bk2,bm2)
    add_test(NAME cosma-test-ddbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 20 -n 30 -k 25 -P 4)

    add_test(NAME cosma-test-dbdb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 10 cosma-test -m 100 -n 100 -k 100 -P 10 -s dm2,bn2,dk2,bm5)
    add_test(NAME cosma-test-dbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 10 cosma-test -m 100 -n 100 -k 100 -P 10)

    add_test(NAME cosma-test-bbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 12 cosma-test -m 100 -n 100 -k 100 -P 12 -s bm2,bn2,bk3)
    add_test(NAME cosma-test-bbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 12 cosma-test -m 100 -n 100 -k 100 -P 12)

    add_test(NAME cosma-test-bdbd-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 100 -n 100 -k 100 -P 4 -s bm2,dk3,bn2,dm3)
    add_test(NAME cosma-test-bdbd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 100 -n 100 -k 100 -P 4)

    add_test(NAME cosma-test-b-prime-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 7 cosma-test -m 100 -n 100 -k 100 -P 7 -s bm7)
    add_test(NAME cosma-test-b-prime COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 7 cosma-test -m 100 -n 100 -k 100 -P 7)

    add_test(NAME cosma-test-dbdbdb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s dm2,bn2,dk2,bm2,dn2,bk2)
    add_test(NAME cosma-test-dbdbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8)

    add_test(NAME cosma-test-dddbbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s dm2,dk2,dn2,bn2,bm2,bk2)

    add_test(NAME cosma-test-dddbbb-reshuffled-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8 -s dk3,dm3,dn3,bk2,bn2,bm2)
    add_test(NAME cosma-test-dddbbb-reshuffled COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8)

    add_test(NAME cosma-test-bbbddd-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s bn2,bm2,bk2,dm2,dk2,dn2)

    add_test(NAME cosma-test-dbdbdb-reshuffled-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s dk2,bm2,dn2,bk2,dm2,bn2)

    add_test(NAME cosma-test-bdbdbd-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s bk2,dm2,bn2,dk2,bm2,dn2)

    add_test(NAME cosma-test-dbdbdb-3-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8 -s dm3,bn2,dk3,bm2,dn3,bk2)
    add_test(NAME cosma-test-dbdbdb-3 COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8)
endif ()









