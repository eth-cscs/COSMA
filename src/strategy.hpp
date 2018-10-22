#ifndef _STRATEGY_H_
#define _STRATEGY_H_

#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <math.h>
#include <limits>
#include <tuple>
#include "options.hpp"
#include "math_utils.hpp"

class Strategy {
public:
    // matrix dimensions
    int m;
    int n;
    int k;
    // number of processors
    size_t P;
    long long memory_limit;
    // beta parameter of gemm
    double beta;
    // stores the divisor in each step of the algorithm
    std::vector<int> divisors;
    // returns m, n or k character depending on 
    // which dimension was split in each step
    std::string split_dimension;
    // describes whether step is DFS (d) or BFS (b) for each step
    std::string step_type;
    // number of steps of the algorithm
    size_t n_steps;
    // if true, MPI will try to relabel ranks such that
    // the ranks which communicate are physically close to each other
    bool topology;
    // if true, one sided communication backend will be used
    // otherwise, two sided communication backend is used
    bool one_sided_communication;
    long long memory_used;
    int n_bfs_steps;
    int n_dfs_steps;

    int n_bfs_steps_before_gemm_a;
    int n_bfs_steps_before_gemm_b;
    int n_bfs_steps_before_gemm_c;

    // constructors
    Strategy();
    // move constructor
    Strategy(Strategy&& other);

    // constructs the Strategy from the command line
    Strategy(const std::string& cmd_line);

    // constructs the Strategy form the command line
    Strategy(int argc, char** argv);

    Strategy(int mm, int nn, int kk, size_t PP,
             std::vector<int>& divs, std::string& dims, std::string& types,
             long long mem_limit = std::numeric_limits<long long>::max(),
             double b = 0.0, bool top = false, bool one_sided = false);

    Strategy(int mm, int nn, int kk, size_t PP, 
            long long mem_limit = std::numeric_limits<long long>::max(),
            double b = 0.0, bool top = false, bool one_sided = false);

    // parses the command line options and initializes the varialbes
    void initialize(const std::string& cmd_line);

    // parses steps if defined manually by the user
    void process_steps(size_t start, const std::string& line);

    // default strategy dividing always the largest dimension in that step
    // if there is enough memory uses BFS step, if not uses DFS step
    void default_strategy();
    // strategy that tries to make each base case as square as possible
    // it always uses all the resources (all P available ranks) but tries to find
    // divm, divn and divk such that divm * divn * divk = P and m/divm = n/divn = k/divk.
    // if there is not enough memory in some step, then DFS step is performed and new
    // divm, divn and divk are found that correspond to the new subproblem.
    void square_strategy();

    void spartition_strategy();

    // token is a triplet e.g. bm3 (denoting BFS (m / 3) step)
    void process_token(const std::string& step_triplet);

    void throw_exception(const std::string& message);

    const bool split_m(size_t i) const;
    const bool split_n(size_t i) const;
    const bool split_k(size_t i) const;

    const bool split_A(size_t i) const;
    const bool split_B(size_t i) const;
    const bool split_C(size_t i) const;

    const bool dfs_step(size_t i) const;
    const bool bfs_step(size_t i) const;

    const int divisor(size_t i) const;
    const int divisor_m(size_t i) const;
    const int divisor_n(size_t i) const;
    const int divisor_k(size_t i) const;

    const int divisor_row(char matrix, size_t i) const;
    const int divisor_col(char matrix, size_t i) const;

    const bool final_step(size_t i) const;
    const int bfs_steps_before_gemm(char label) const;

    static long long initial_memory(long long m, long long n, long long k, int P);
    static long long required_memory(Strategy& strategy);

    // checks if the strategy is well-defined
    void check_if_valid();

    friend std::ostream& operator<<(std::ostream& os, const Strategy& other);
};
#endif
