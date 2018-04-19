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

add_lib_executable(carma-test
  carma
  ${carma_deps})

target_link_libraries(carma-test semiprof)

################
#  unit gtests #
################
set(TEST_SOURCES test_mapper.cpp unit_test.cpp)

set(target unit_test.exe)
add_executable(unit_test.exe ${TEST_SOURCES})
target_link_libraries(${target} gtest)
target_link_libraries(${target} carma)
add_test(layout-and-mapper-tests unit_test.exe)

# debug with
# mpirun -np 1 xterm -e lldb -o run -- ./carma-test  ...

if (MPI_FOUND)
# test the correctness of carma
    add_test(NAME carma-test-bbdd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 carma-test -m 30 -n 35 -k 40 -P 4 -s bm2,bn2,dk2,dm2)

    add_test(NAME carma-test-bdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 carma-test -m 8 -n 4 -k 2 -P 4 -s bm2,dm2,bn2)

    add_test(NAME carma-test-ddb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 2 carma-test -m 8 -n 8 -k 2 -P 2 -s dm2,dm2,bn2)

    add_test(NAME carma-test-bb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 carma-test -m 16 -n 4 -k 4 -P 4 -s bm2,bm2)

    add_test(NAME carma-test-db COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 3 carma-test -m 20 -n 20 -k 20 -P 3 -s dk2,bm3)

    add_test(NAME carma-test-bbbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 carma-test -m 16 -n 16 -k 16 -P 16 -s bm2,bn2,bk2,bm2)

    add_test(NAME carma-test-dddd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 1 carma-test -m 16 -n 16 -k 16 -P 1 -s dm2,dn2,dm2,dn2)

    add_test(NAME carma-test-ddbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 carma-test -m 20 -n 30 -k 25 -P 4 -s dm2,dn2,bk2,bm2)

    add_test(NAME carma-test-dbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 10 carma-test -m 100 -n 100 -k 100 -P 10 -s dm2,bn2,dk2,bm5)

    add_test(NAME carma-test-bbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 12 carma-test -m 100 -n 100 -k 100 -P 12 -s bm2,bn2,bk3)

    add_test(NAME carma-test-bdbd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 carma-test -m 100 -n 100 -k 100 -P 4 -s bm2,dk3,bn2,dm3)

    add_test(NAME carma-test-b-prime COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 7 carma-test -m 100 -n 100 -k 100 -P 7 -s bm7)

    add_test(NAME carma-test-dbdbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 100 -n 100 -k 100 -P 8 -s dm2,bn2,dk2,bm2,dn2,bk2)

    add_test(NAME carma-test-dddbbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 100 -n 100 -k 100 -P 8 -s dm2,dk2,dn2,bn2,bm2,bk2)

    add_test(NAME carma-test-dddbbb-reshuffled COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 200 -n 200 -k 200 -P 8 -s dk3,dm3,dn3,bk2,bn2,bm2)

    add_test(NAME carma-test-bbbddd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 100 -n 100 -k 100 -P 8 -s bn2,bm2,bk2,dm2,dk2,dn2)

    add_test(NAME carma-test-dbdbdb-reshuffled COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 100 -n 100 -k 100 -P 8 -s dk2,bm2,dn2,bk2,dm2,bn2)

    add_test(NAME carma-test-bdbdbd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 100 -n 100 -k 100 -P 8 -s bk2,dm2,bn2,dk2,bm2,dn2)

    add_test(NAME carma-test-dbdbdb-3 COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 carma-test -m 200 -n 200 -k 200 -P 8 -s dm3,bn2,dk3,bm2,dn3,bk2)
endif ()









