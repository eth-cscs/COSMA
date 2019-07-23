#include <cosma/blas.hpp>
#include <cosma/multiply.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

template <typename Real, typename Allocator>
void fill_matrix(std::vector<Real, Allocator> &data) {
    std::random_device dev;                        // seed
    std::mt19937 rng(dev());                       // generator
    std::uniform_real_distribution<Real> dist(1.); // distribution

    for (auto &el : data) {
        el = Real{dist(rng)};
    }
}

template <typename Real, typename Allocator>
void fill_matrix(std::vector<std::complex<Real>, Allocator> &data) {
    std::random_device dev;                        // seed
    std::mt19937 rng(dev());                       // generator
    std::uniform_real_distribution<Real> dist(1.); // distribution

    for (auto &el : data) {
        el = std::complex<Real>{dist(rng), dist(rng)};
    }
}

template <typename Scalar>
void run(cosma::Strategy &strategy,
         cosma::context &ctx,
         std::string scalar_str) {
    using seconds_t = std::chrono::duration<double>;
    using clock_t = std::chrono::high_resolution_clock;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);

    constexpr int num_matmuls = 1;
    std::stringstream times_str;
    times_str << scalar_str << " : ";
    for (int i = 0; i < num_matmuls; ++i) {
        cosma::CosmaMatrix<Scalar> A('A', strategy, rank);
        cosma::CosmaMatrix<Scalar> B('B', strategy, rank);
        cosma::CosmaMatrix<Scalar> C('C', strategy, rank);
        constexpr auto alpha = Scalar{1};
        constexpr auto beta = Scalar{0};

        fill_matrix(A.matrix());
        fill_matrix(B.matrix());

        auto start = clock_t::now();
        multiply(ctx, A, B, C, strategy, comm, alpha, beta);
        auto end = clock_t::now();
        times_str << seconds_t(end - start).count() << "s ";
    }

    if (rank == 0) {
        std::cout << times_str.str() << '\n';
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    cosma::Strategy strategy(argc, argv);
    auto ctx = cosma::make_context();

    run<float>(strategy, ctx, "Float");
    run<double>(strategy, ctx, "Double");
    run<std::complex<float>>(strategy, ctx, "Complex Float");
    run<std::complex<double>>(strategy, ctx, "Complex Double");

    MPI_Finalize();
}
