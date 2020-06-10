#include <cosma/multiply.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "../utils/parse_strategy.hpp"

using namespace cosma;

template <typename T>
void fill_int(T* ptr, size_t size) {
    for (unsigned i = 0u; i < size; ++i) {
        ptr[i] = 10*drand48();
    }
}

// Reads an environment variable `n_iter`
//
int get_n_iter() {
    auto env = std::getenv("n_iter");
    return env == nullptr ? 1 : std::atoi(env);
}

void output_matrix(CosmaMatrix<double> &M, int rank) {
    std::string local = M.which_matrix() + std::to_string(rank) + ".txt";
    std::ofstream local_file(local);
    local_file << M << std::endl;
    local_file.close();
}

long run(const Strategy &s, MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Declare A,B and C COSMA matrices objects
    CosmaMatrix<double> A('A', s, rank);
    CosmaMatrix<double> B('B', s, rank);
    CosmaMatrix<double> C('C', s, rank);

    double alpha = 1;
    double beta = 0;

    // fill the matrices with random data
    srand48(rank);
    fill_int(A.matrix_pointer(), A.matrix_size());
    fill_int(B.matrix_pointer(), B.matrix_size());

    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    multiply(A, B, C, s, comm, alpha, beta);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const Strategy& strategy = parse_strategy<double>(argc, argv);

    if (rank == 0) {
        std::cout << "Strategy = " << strategy << std::endl;
    }

    int n_iter = get_n_iter();
    std::vector<long> times;
    for (int i = 0; i < n_iter; ++i) {
        long t_run = 0;
        try {
            t_run = run(strategy);
        } catch (const std::exception& e) {
            int flag = 0;
            MPI_Finalized(&flag);
            if (!flag) {
                MPI_Abort(MPI_COMM_WORLD, -1);
                MPI_Finalize();
            }
            return 0;
        }
        times.push_back(t_run);
    }
    std::sort(times.begin(), times.end());

    if (rank == 0) {
        std::cout << "COSMA TIMES [ms] = ";
        for (auto &time : times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
