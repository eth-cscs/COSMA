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

add_test(layout-and-mapper-tests unit_test.exe)

if (MPI_FOUND)
# test the correctness of cosma
    add_test(NAME cosma-test-buffers-test COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 4 -n 4 -k 4 -P 4 -s sm2,pn2,pn2)

    add_test(NAME cosma-test-30-35-40 COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 30 -n 35 -k 40 -P 4)

    add_test(NAME cosma-test-bdb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 8 -n 4 -k 2 -P 4 -s pm2,sm2,pn2)
    add_test(NAME cosma-test-bdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 8 -n 4 -k 2 -P 4)

    add_test(NAME cosma-test-ddb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 2 cosma-test -m 8 -n 8 -k 2 -P 2 -s sm2,sm2,pn2)
    add_test(NAME cosma-test-ddb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 2 cosma-test -m 8 -n 8 -k 2 -P 2)

    add_test(NAME cosma-test-bb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 16 -n 4 -k 4 -P 4 -s pm2,pm2)
    add_test(NAME cosma-test-bb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 16 -n 4 -k 4 -P 4)

    add_test(NAME cosma-test-db-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 3 cosma-test -m 20 -n 20 -k 20 -P 3 -s sk2,pm3)
    add_test(NAME cosma-test-db COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 3 cosma-test -m 20 -n 20 -k 20 -P 3)

    add_test(NAME cosma-test-bbbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 cosma-test -m 16 -n 16 -k 16 -P 16 -s pm2,pn2,pk2,pm2)
    add_test(NAME cosma-test-bbbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 16 cosma-test -m 16 -n 16 -k 16 -P 16)

    # add_test(NAME cosma-test-dddd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
    #    ${MPIEXEC_NUMPROC_FLAG} 1 cosma-test -m 16 -n 16 -k 16 -P 1)

    add_test(NAME cosma-test-ddbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 20 -n 30 -k 25 -P 4 -s sm2,sn2,pk2,pm2)
    add_test(NAME cosma-test-ddbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 20 -n 30 -k 25 -P 4)

    add_test(NAME cosma-test-dbdb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 10 cosma-test -m 100 -n 100 -k 100 -P 10 -s sm2,pn2,sk2,pm5)
    add_test(NAME cosma-test-dbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 10 cosma-test -m 100 -n 100 -k 100 -P 10)

    add_test(NAME cosma-test-dbdb-445 COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 4 -n 4 -k 5 -P 4 -s sm2,pn2,sk2,pm2)

    add_test(NAME cosma-test-bbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 12 cosma-test -m 100 -n 100 -k 100 -P 12 -s pm2,pn2,pk3)
    add_test(NAME cosma-test-bbb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 12 cosma-test -m 100 -n 100 -k 100 -P 12)

    add_test(NAME cosma-test-bdbd COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 4 cosma-test -m 100 -n 100 -k 100 -P 4)

    add_test(NAME cosma-test-b-prime-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 7 cosma-test -m 100 -n 100 -k 100 -P 7 -s pm7)
    add_test(NAME cosma-test-b-prime COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 7 cosma-test -m 100 -n 100 -k 100 -P 7)

    add_test(NAME cosma-test-dbdbdb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s sm2,pn2,sk2,pm2,sn2,pk2)
    add_test(NAME cosma-test-dbdbdb COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8)

    add_test(NAME cosma-test-dddbbb-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s sm2,sk2,sn2,pn2,pm2,pk2)

    add_test(NAME cosma-test-dddbbb-reshuffled-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8 -s sk3,sm3,sn3,pk2,pn2,pm2)
    add_test(NAME cosma-test-dddbbb-reshuffled COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8)

    add_test(NAME cosma-test-dbdbdb-reshuffled-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 100 -n 100 -k 100 -P 8 -s sk2,pm2,sn2,pk2,sm2,pn2)

    add_test(NAME cosma-test-dbdbdb-3-s COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8 -s sm3,pn2,sk3,pm2,sn3,pk2)
    add_test(NAME cosma-test-dbdbdb-3 COMMAND ${MPIEXEC} ${OVERSUBSCRIBE}
        ${MPIEXEC_NUMPROC_FLAG} 8 cosma-test -m 200 -n 200 -k 200 -P 8)
endif ()









