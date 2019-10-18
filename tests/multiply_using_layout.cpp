#include <cosma/local_multiply.hpp>
#include <cosma/multiply.hpp>

#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>
#include <mpi.h>

#include <cmath>
#include <limits>

template <typename scalar>
void fill_matrix(cosma::CosmaMatrix<scalar> &M) {
    for (int idx = 0; idx < M.matrix_size(); ++idx) {
        M[idx] = std::sin(idx);
    }
};

// !!! [NOTE] The test depends on correct implementation of `multiply()`.
//
TEST(MultiplyUsingLayout, ) {
    using scalar_t = double;
    constexpr int nprocs = 4;
    constexpr int m = 20;
    constexpr int n = 20;
    constexpr int k = 80;
    constexpr scalar_t alpha = 1;
    constexpr scalar_t beta = 0;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    cosma::Strategy strategy(m, n, k, nprocs);

    cosma::CosmaMatrix<scalar_t> A('A', strategy, rank);
    cosma::CosmaMatrix<scalar_t> B('B', strategy, rank);
    cosma::CosmaMatrix<scalar_t> C('C', strategy, rank);

    fill_matrix(A);
    fill_matrix(B);

    auto A_grid = A.get_grid_layout();
    auto B_grid = B.get_grid_layout();
    auto C_grid = C.get_grid_layout();
    cosma::multiply_using_layout(A_grid, B_grid, C_grid, alpha, beta, comm);

    cosma::CosmaMatrix<scalar_t> C_act('C', strategy, rank);
    cosma::multiply(A, B, C_act, strategy, comm, alpha, beta);

    // ----- Checks for data integrity

    scalar_t *C_data = C.buffer_ptr();
    scalar_t *C_act_data = C_act.buffer_ptr();

    // Check if sizes match.
    //
    ASSERT_EQ(C.buffer_size(), C_act.buffer_size());

    // Check if data elements match. Fail if an element doesn't match.
    //
    for (int i = 0; i < C_act.buffer_size(); ++i) {
        ASSERT_DOUBLE_EQ(C_data[i], C_act_data[i]);
    }
}
