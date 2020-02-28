#include "../utils/parse_strategy.hpp"
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

template <typename T>
void fill_matrix(T* ptr, size_t size) {
    std::random_device dev;                        // seed
    std::mt19937 rng(dev());                       // generator
    std::uniform_real_distribution<T> dist(1.); // distribution

    for (unsigned i = 0; i < size; ++i) {
        ptr[i] = T{dist(rng)};
    }
}

template <typename T>
void fill_matrix(std::complex<T>* ptr, size_t size) {
    std::random_device dev;                        // seed
    std::mt19937 rng(dev());                       // generator
    std::uniform_real_distribution<T> dist(1.); // distribution

    for (unsigned i = 0; i < size; ++i) {
        ptr[i] = std::complex<T>{dist(rng), dist(rng)};
    }
}

template <typename Scalar>
void run(const cosma::Strategy &strategy,
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

        fill_matrix(A.matrix_pointer(), A.matrix_size());
        fill_matrix(B.matrix_pointer(), B.matrix_size());

        auto start = clock_t::now();
        multiply(A, B, C, strategy, comm, alpha, beta);
        auto end = clock_t::now();
        times_str << seconds_t(end - start).count() << "s ";
    }

    if (rank == 0) {
        std::cout << times_str.str() << '\n';
    }
}

int main(int argc, char **argv) {
    using zfloat = std::complex<float>;
    using zdouble = std::complex<double>;

    MPI_Init(&argc, &argv);

    const cosma::Strategy& strategy = parse_strategy(argc, argv);

    auto cxt_f = cosma::make_context<float>();
    run<float>(strategy, "Float");

    auto cxt_d = cosma::make_context<double>();
    run<double>(strategy, "Double");

    auto cxt_zf = cosma::make_context<zfloat>();
    run<zfloat>(strategy, "Complex Float");

    auto cxt_zd = cosma::make_context<zdouble>();
    run<zdouble>(strategy, "Complex Double");

    MPI_Finalize();
}
