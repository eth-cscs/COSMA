cmake_minimum_required(VERSION 3.5)

project(tests)

set(BIN_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TEST_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

####################
#  Lib Dependancy  #
####################

#include_directories( "${PRJ_ROOT_DIR}/tests" )
#include_directories( "${PRJ_ROOT_DIR}/profiler" )
include_directories( "${PRJ_ROOT_DIR}/src" )

################
#  Build test  #
################

add_lib_executable(Carma-test
  carma
  ${carma_deps})

################
#  unit gtests #
################
set(TEST_SOURCES
test_mapper.cpp
unit_test.cpp)

set(target unit_test.exe)
add_executable(unit_test.exe ${TEST_SOURCES})
target_link_libraries(${target} gtest)
target_link_libraries(${target} carma)
add_test(layout-and-mapper-tests unit_test.exe)

# debug with
# mpirun -np 1 xterm -e lldb -o run -- ./Carma-lib-test  -m 16 -n 16 -k 16 -r 4 -p dddd -d 211121112211

if (MPI_FOUND)
# test the correctness of carma
    add_test(NAME Carma-test-bbdd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 Carma-test -m 30 -n 35 -k 40 -r 4 -p bbdd -d 211121112211)

    add_test(NAME Carma-test-bdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 Carma-test -m 8 -n 4 -k 2 -r 3 -p bdb -d 211211121)

    add_test(NAME Carma-test-ddb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 2 Carma-test -m 8 -n 8 -k 2 -r 3 -p ddb -d 211211121)

    add_test(NAME Carma-test-bb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 Carma-test -m 16 -n 4 -k 4 -r 2 -p bb -d 211211)

    add_test(NAME Carma-test-db COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 3 Carma-test -m 20 -n 20 -k 20 -r 2 -p db -d 112311)

    add_test(NAME Carma-test-bbbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 Carma-test -m 16 -n 16 -k 16 -r 4 -p bbbb -d 211121112211)

    add_test(NAME Carma-test-dddd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 1 Carma-test -m 16 -n 16 -k 16 -r 4 -p dddd -d 211121211121)

    add_test(NAME Carma-test-ddbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 Carma-test -m 20 -n 30 -k 25 -r 4 -p ddbb -d 211121112211)

    add_test(NAME Carma-test-dbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 10 Carma-test -m 100 -n 100 -k 100 -r 4 -p dbdb -d 211121112511)

    add_test(NAME Carma-test-bbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 12 Carma-test -m 100 -n 100 -k 100 -r 3 -p bbb -d 211121113)

    add_test(NAME Carma-test-bdbd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 Carma-test -m 100 -n 100 -k 100 -r 4 -p bdbd -d 211113121311)

    add_test(NAME Carma-test-b-prime COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 7 Carma-test -m 100 -n 100 -k 100 -r 1 -p b -d 711)

    add_test(NAME Carma-test-dbdbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 100 -n 100 -k 100 -r 6 -p dbdbdb -d 211121112211121112)

    add_test(NAME Carma-test-dddbbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 100 -n 100 -k 100 -r 6 -p dddbbb -d 211112121121211112)

    add_test(NAME Carma-test-dddbbb-reshuffled COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 200 -n 200 -k 200 -r 6 -p dddbbb -d 113311131112121211)

    add_test(NAME Carma-test-bbbddd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 100 -n 100 -k 100 -r 6 -p bbbddd -d 121211112211112121)

    add_test(NAME Carma-test-dbdbdb-reshuffled COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 100 -n 100 -k 100 -r 6 -p dbdbdb -d 112211121112211121)

    add_test(NAME Carma-test-bdbdbd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 100 -n 100 -k 100 -r 6 -p bdbdbd -d 112211121112211121)

    add_test(NAME Carma-test-dbdbdb-3 COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 Carma-test -m 200 -n 200 -k 200 -r 6 -p dbdbdb -d 311121113211131112)
endif ()









