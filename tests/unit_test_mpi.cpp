#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // init gtest
    ::testing::InitGoogleTest(&argc, argv);

    // new test environment (makes a copy of MPI_COMM_WORLD)
    ::testing::AddGlobalTestEnvironment(new gtest_mpi::MPITestEnvironment());

    auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();

    // replace the default listener with the custom one
    delete test_listeners.Release(test_listeners.default_result_printer());
    test_listeners.Append(new gtest_mpi::PrettyMPIUnitTestResultPrinter());

    // run all tests
    auto exit_code = RUN_ALL_TESTS();

    MPI_Finalize();

    return exit_code;
}
