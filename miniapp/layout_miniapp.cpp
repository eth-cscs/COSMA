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
#include <stdlib.h>
#include <vector>


#include "../utils/parse_strategy.hpp"
#include "../utils/cosma_utils.hpp"

#include <cxxopts.hpp>

using namespace cosma;

int main(int argc, char **argv) {
    cxxopts::Options options("NATIVE COSMA LAYOUT MINIAPP", 
        "A miniapp showing the native COSMA data layout for computing C=A*B, where dim(A)=m*k, dim(B)=k*n, dim(C)=m*n");
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
        ("P,n_ranks",
            "number of MPI ranks.", 
            cxxopts::value<int>()->default_value("1"))
        ("s,steps", 
            "Division steps that the algorithm should perform. Can be empty.",
            cxxopts::value<std::vector<std::string>>()->default_value(""))
        ("h,help", "Print usage.")
        ;

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto m = result["m_dim"].as<int>();
    auto n = result["n_dim"].as<int>();
    auto k = result["k_dim"].as<int>();
    auto P = result["n_ranks"].as<int>();
    auto steps = result["steps"].as<std::vector<std::string>>();

    // prevent the optimization that might reduce the number of ranks
    std::string set_min_local_size = "COSMA_MIN_LOCAL_DIMENSION=1";
    putenv(&set_min_local_size[0]);

    // specified by the environment variable COSMA_CPU_MAX_MEMORY
    long long memory_limit = cosma::get_cpu_max_memory<double>();

    // specified by the env var COSMA_OVERLAP_COMM_AND_COMP
    bool overlap_comm_and_comp = cosma::get_overlap_comm_and_comp();
    const Strategy& strategy = parse_strategy(m, n, k, P,
                                              steps,
                                              memory_limit,
                                              overlap_comm_and_comp);

    std::cout << "Strategy = " << strategy << std::endl;

    int rank = 0;

    // Declare A,B and C COSMA matrices objects
    CosmaMatrix<double> A('A', strategy, rank);
    CosmaMatrix<double> B('B', strategy, rank);
    CosmaMatrix<double> C('C', strategy, rank);

    auto A_layout = A.get_grid_layout();
    auto B_layout = B.get_grid_layout();
    auto C_layout = C.get_grid_layout();

    std::cout << "A matrix layout =\n" << A_layout.grid << std::endl;
    std::cout << "B matrix layout =\n" << B_layout.grid << std::endl;
    std::cout << "C matrix layout =\n" << C_layout.grid << std::endl;

    /*
    if (std::max(std::max(m, n), k) < 20) {
        std::cout << "\n===============================\n\n" << std::endl;
        std::cout << "Visually, the matrices is distributed among ranks as follows:\n\n";
        std::cout << "Matrix A:\n";
        for (unsigned bi = 0; bi < A_layout.grid.num_blocks_row(); ++bi) {
            for (unsigned i = A_layout.grid.rows_interval(bi).start; 
                          i < A_layout.grid.rows_interval(bi).end; 
                          ++i) {
                for (unsigned bj = 0; bj < A_layout.grid.num_blocks_col(); ++bj) {
                    auto owner = A_layout.grid.owner(bi, bj);

                    for (unsigned j = A_layout.grid.cols_interval(bj).start; 
                                  j < A_layout.grid.cols_interval(bj).end; 
                                  ++j) {
                        std::cout << owner << "\t";
                    }
                }
                std::cout << "\n";
            }
        }
    }
    */

    return 0;
}
