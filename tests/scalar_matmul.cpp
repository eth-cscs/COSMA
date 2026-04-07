#include <gtest/gtest.h>
#include <gtest_mpi/gtest_mpi.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "../utils/cosma_utils.hpp"

template <typename Scalar>
void test_matmul() {
    constexpr int m = 100;
    constexpr int n = 100;
    constexpr int k = 100;
    constexpr int P = 8;
    std::string step_types = "spspsp";
    std::string dims = "mnkmnk";
    std::vector<int> divs = {2, 2, 2, 2, 2, 2};

    auto comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    Strategy strategy(m, n, k, P, divs, dims, step_types);

    if (rank == 0) {
        std::cout << "Strategy = " << strategy << std::endl;
    }

    auto ctx = cosma::make_context<Scalar>();

    // first run without overlapping communication and computation
    bool no_overlap = test_cosma<Scalar>(strategy, ctx, comm, 1e-2, 0);
    ASSERT_TRUE(no_overlap);

    // enable the ovelap of comm and comp
    strategy.enable_overlapping_comm_and_comp();

    // then run with the overlap of communication and computation
    bool with_overlap = test_cosma<Scalar>(strategy, ctx, comm, 1e-2, 1);
    ASSERT_TRUE(with_overlap);
}

TEST(Multiply, Float) { test_matmul<float>(); }

TEST(Multiply, Double) { test_matmul<double>(); }

TEST(Multiply, ComplexFloat) { test_matmul<std::complex<float>>(); }

TEST(Multiply, ComplexDouble) { test_matmul<std::complex<double>>(); }

// Test: beta=0 with uninitialized C must not produce NaN.
// This covers the BLAS spec requirement that C is not read when beta=0.
// Regression test for split_k accumulation bug where iterations K>0
// used beta=1 on uninitialized pool memory.
template <typename Scalar>
void test_beta_zero_uninitialized_c() {
    constexpr int m = 64;
    constexpr int n = 64;
    constexpr int k = 64;

    std::vector<Scalar> A(m * k);
    std::vector<Scalar> B(k * n);
    std::vector<Scalar> C(m * n);

    // Fill A, B with small values
    for (int i = 0; i < m * k; ++i) A[i] = Scalar{0.01} * (i % 17 + 1);
    for (int i = 0; i < k * n; ++i) B[i] = Scalar{0.01} * (i % 13 + 1);

    // Fill C with NaN — simulates uninitialized memory pool
    Scalar nan_val = std::numeric_limits<Scalar>::quiet_NaN();
    std::fill(C.begin(), C.end(), nan_val);

    // beta=0: BLAS spec says C content must be ignored
    cosma::local_multiply_cpu(A.data(), B.data(), C.data(),
                              m, n, k, Scalar{1}, Scalar{0});

    // Verify no NaN in result
    for (int i = 0; i < m * n; ++i) {
        ASSERT_FALSE(std::isnan(C[i]))
            << "NaN at index " << i << ": beta=0 should ignore C content";
    }

    // Verify result is correct: C = 1*A*B + 0*C = A*B
    // Spot-check C[0] = sum(A[0,k] * B[k,0]) for k=0..63
    Scalar expected = Scalar{0};
    for (int ki = 0; ki < k; ++ki) {
        expected += A[ki * m] * B[ki];  // col-major: A(0,ki) = A[ki*m], B(ki,0) = B[ki]
    }
    ASSERT_NEAR(double(C[0]), double(expected), 1e-4)
        << "C[0] incorrect: expected " << expected << " got " << C[0];
}

TEST(BetaZero, FloatUninitialized) { test_beta_zero_uninitialized_c<float>(); }
TEST(BetaZero, DoubleUninitialized) { test_beta_zero_uninitialized_c<double>(); }
