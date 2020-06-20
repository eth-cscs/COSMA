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

template <typename T>
void output_matrix(CosmaMatrix<T> &M, int rank) {
    std::string local = M.which_matrix() + std::to_string(rank) + ".txt";
    std::ofstream local_file(local);
    local_file << M << std::endl;
    local_file.close();
}

template <typename T>
long run(const int m, const int n, const int k, 
         const std::vector<std::string>& steps, 
         MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // specified by the environment variable COSMA_CPU_MAX_MEMORY
    long long memory_limit = cosma::get_cpu_max_memory<T>();

    // specified by the env var COSMA_OVERLAP_COMM_AND_COMP
    bool overlap_comm_and_comp = cosma::get_overlap_comm_and_comp();

    const Strategy& strategy = parse_strategy(m, n, k,
                                              steps,
                                              memory_limit,
                                              overlap_comm_and_comp);

    if (rank == 0) {
        std::cout << "Strategy = " << strategy << std::endl;
    }

    // Declare A,B and C COSMA matrices objects
    CosmaMatrix<T> A('A', s, rank);
    CosmaMatrix<T> B('B', s, rank);
    CosmaMatrix<T> C('C', s, rank);

    T alpha{1};
    T beta{0};

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

    cxxopts::Options options("COSMA MINIAPP", 
        "A miniapp computing: `C=A*B, where dim(A)=m x k, dim(B)=k x n, dim(C)=m x n");
    options.add_options()
        ("m,m_dim",
            "number of rows of A and C.", 
            cxxopts::value<int>()->default_value("1000"))
        ("n,n_dim",
            "number of columns of B and C.",
            cxxopts::value<int>()->default_value("1000"))
        ("k,k_dim",
            "number of columns of A and rows of B.", 
            cxxopts::value<int>()->default_value("1000"))
        ("s,steps", 
            "Division steps that the algorithm should perform.",
            cxxopts::value<std::vector<std::string>>())
        ("r,n_rep",
            "number of repetitions.", 
            cxxopts::value<int>()->default_value("2"))
        ("t,type",
            "data type of matrix entries.",
            cxxopts::value<std::string>()->default_value("double"))
        ("h,help", "Print usage.")

    auto result = options.parse(argc, argv);

    auto m = result["m_dim"].as<int>();
    auto n = result["n_dim"].as<int>();
    auto k = result["k_dim"].as<int>();
    auto steps = result["steps"].as<std::vector<std::string>>();
    auto n_rep = result["n_rep"].as<int>();
    auto type = result["type"].as<std::string>();

    std::vector<long> times;
    for (int i = 0; i < n_rep; ++i) {
        long t_run = 0;
        try {
            if (type == "double") {
                t_run = run<double>(m, n, k, steps, comm);
            } else if (type == "float") {
                t_run = run<float>(m, n, k, steps, comm);
            } else if (type = "zdouble") {
                t_run = run<std::complex<double>>(m, n, k, steps, comm);
            } else if (type = "zfloat") {
                t_run = run<std::complex<float>>(m, n, k, steps, comm);
            } else {
                throw std::runtime_error("COSMA(cosma_miniapp): unknown data type of matrix entries.");
            }
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
