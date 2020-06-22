/*
Simulates the algorithm (without actually computing the matrix multiplication)
 * in order to get the total volume of the communication, the maximum volume of computation
 * done in a single branch and the maximum required buffer size that the algorithm requires.
 */
#include "../utils/parse_strategy.hpp"
#include <cosma/statistics.hpp>
#include <cxxopts.hpp>

#include <iostream>

using namespace cosma;

int main( int argc, char **argv ) {
    cxxopts::Options options("COSMA STATISTICS",
                             "A miniapp computing communication volume \
                             and local multiplication sizes. dim(A)=m x k, dim(B)=k x n; dim(C)=m x n.");
    
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
        ("P,n_proc",
            "Number of MPI ranks.", 
            cxxopts::value<int>()->default_value("1"))
        ("s,steps", 
            "Division steps that the algorithm should perform.",
            cxxopts::value<std::vector<std::string>>())
        ("h,help", "Print usage.")
    ;

    auto result = options.parse(argc, argv);

    auto m = result["m_dim"].as<int>();
    auto n = result["n_dim"].as<int>();
    auto k = result["k_dim"].as<int>();
    auto P = result["n_proc"].as<int>();
    auto steps = result["steps"].as<std::vector<std::string>>();
    auto type = result["type"].as<std::string>();

    bool overlap_comm_and_comp = cosma::get_overlap_comm_and_comp();
    long long memory_limit = cosma::get_cpu_max_memory<double>();

    const Strategy& strategy = parse_strategy(m, n, k, P,
                                              steps,
                                              memory_limit,
                                              overlap_comm_and_comp);

    int n_rep = 1;
    multiply(strategy, n_rep);

    return 0;
}
