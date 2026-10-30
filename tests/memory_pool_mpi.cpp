#include "../utils/cosma_utils.hpp"

#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>

// Regression test for cosma_context::free_memory_pool(): releasing the
// pool between two independent multiply() calls (each of which builds
// its own fresh CosmaMatrix/Buffer objects, as test_cosma() does) must
// not corrupt the pool nor the results of the following multiplication.
TEST(MemoryPool, FreeBetweenMultipliesIsSafe) {
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    Strategy strategy(200, 200, 200, P);
    auto ctx = cosma::make_context<double>();

    bool first_ok = test_cosma<double>(strategy, ctx, MPI_COMM_WORLD, 1e-8, 0);
    ASSERT_TRUE(first_ok);

    ctx->free_memory_pool();

    bool second_ok = test_cosma<double>(strategy, ctx, MPI_COMM_WORLD, 1e-8, 1);
    ASSERT_TRUE(second_ok);
}
