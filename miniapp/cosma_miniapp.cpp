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
#include "../utils/cosma_utils.hpp"

#include <cxxopts.hpp>

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
bool run(const int m, const int n, const int k, 
         const std::vector<std::string>& steps, 
         long& timing, const bool test_correctness,
         MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // specified by the environment variable COSMA_CPU_MAX_MEMORY
    long long memory_limit = cosma::get_cpu_max_memory<T>();

    if (!test_correctness) {
        // specified by the env var COSMA_OVERLAP_COMM_AND_COMP
        bool overlap_comm_and_comp = cosma::get_overlap_comm_and_comp();
        const Strategy& strategy = parse_strategy(m, n, k, size,
                                                  steps,
                                                  memory_limit,
                                                  overlap_comm_and_comp);

        if (rank == 0) {
            std::cout << "Strategy = " << strategy << std::endl;
        }

        // Declare A,B and C COSMA matrices objects
        CosmaMatrix<T> A('A', strategy, rank);
        CosmaMatrix<T> B('B', strategy, rank);
        CosmaMatrix<T> C('C', strategy, rank);

        T alpha{1};
        T beta{0};

        // fill the matrices with random data
        srand48(rank);
        fill_int(A.matrix_pointer(), A.matrix_size());
        fill_int(B.matrix_pointer(), B.matrix_size());

        MPI_Barrier(comm);
        auto start = std::chrono::steady_clock::now();
        multiply(A, B, C, strategy, comm, alpha, beta);
        MPI_Barrier(comm);
        auto end = std::chrono::steady_clock::now();

        timing 
            = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

        return true;
    } else {
        // specified by the env var COSMA_OVERLAP_COMM_AND_COMP
        const Strategy& strategy_no_overlap = parse_strategy(m, n, k, size,
                                                  steps,
                                                  memory_limit,
                                                  false);
        const Strategy& strategy_with_overlap = parse_strategy(m, n, k, size,
                                                  steps,
                                                  memory_limit,
                                                  true);
        if (rank == 0) {
            std::cout << "Strategy = " << strategy_no_overlap << std::endl;
        }

        auto ctx = cosma::make_context<T>();

        // first run without overlapping communication and computation
        bool isOK = test_cosma<T>(strategy_no_overlap, ctx, comm);
        // then run with the overlap of communication and computation
        isOK = isOK && test_cosma<T>(strategy_with_overlap, ctx, comm);

        return rank == 0 ? isOK : true;
    }
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
        ("test",
            "test the result correctness.",
            cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage.")
        ;

    auto result = options.parse(argc, argv);

    auto m = result["m_dim"].as<int>();
    auto n = result["n_dim"].as<int>();
    auto k = result["k_dim"].as<int>();
    auto steps = result["steps"].as<std::vector<std::string>>();
    auto n_rep = result["n_rep"].as<int>();
    auto type = result["type"].as<std::string>();
    bool test_correctness = result["test"].as<bool>();

    bool result_correct = true;

    std::vector<long> times;
    for (int i = 0; i < n_rep; ++i) {
        long t_run = 0;
        try {
            if (type == "double") {
                result_correct = 
                run<double>(m, n, k, steps, 
                            t_run, test_correctness, MPI_COMM_WORLD);
            } else if (type == "float") {
                result_correct = 
                run<float>(m, n, k, steps, 
                           t_run, test_correctness, MPI_COMM_WORLD);
            } else if (type == "zdouble") {
                result_correct = 
                run<std::complex<double>>(m, n, k, steps, 
                                          t_run, test_correctness, MPI_COMM_WORLD);
            } else if (type == "zfloat") {
                result_correct = 
                run<std::complex<float>>(m, n, k, steps, 
                                         t_run, test_correctness, MPI_COMM_WORLD);
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

    if (!test_correctness && rank == 0) {
        std::cout << "COSMA TIMES [ms] = ";
        for (auto &time : times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    if (test_correctness && rank == 0) {
        std::string yes_no = result_correct ? "" : " NOT";
        std::cout << "Result is" << yes_no << " CORRECT!" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
