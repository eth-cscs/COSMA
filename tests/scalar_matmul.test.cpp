#include <cosma_run.hpp>

#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>

#include <string>

template <typename Scalar>
void test_matmul() {
    constexpr int m = 100;
    constexpr int n = 100;
    constexpr int k = 100;
    constexpr int P = 8;
    std::string steps = "-s sm2,pn2,sk2,pm2,sn2,pk2";

    auto comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    auto ctx = cosma::make_context();

    Strategy strategy(m, n, k, P, steps);

    if (rank == 0) {
        std::cout << "Strategy = " << strategy << std::endl;
    }

    // first run without overlapping communication and computation
    bool no_overlap = run<Scalar>(strategy, ctx, comm, false);
    ASSERT_TRUE(no_overlap);

    // wait for no-overlap to finish
    MPI_Barrier(comm);

    // then run with the overlap of communication and computation
    bool with_overlap = run<Scalar>(strategy, ctx, comm, true);
    ASSERT_TRUE(with_overlap);
}

TEST(Multiply, Float) { test_matmul<float>(); }

TEST(Multiply, Double) { test_matmul<double>(); }

TEST(Multiply, ComplexFloat) { test_matmul<std::complex<float>>(); }

TEST(Multiply, ComplexDouble) { test_matmul<std::complex<double>>(); }
