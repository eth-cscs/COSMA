#ifndef _STRATEGY_H_
#define _STRATEGY_H_

#include <stdexcept>
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <iostream>
#include <math.h>
#include <limits>
#include <tuple>

class Strategy {
public:
    // matrix dimensions
    int m;
    int n;
    int k;
    // number of processors
    size_t P;
    // number of steps of the algorithm
    size_t n_steps;
    // stores the divisor in each step of the algorithm
    std::vector<int> divisors;
    // returns m, n or k character depending on 
    // which dimension was split in each step
    std::string split_dimension;
    // describes whether step is DFS (d) or BFS (b) for each step
    std::string step_type;
    // if true, MPI will try to relabel ranks such that
    // the ranks which communicate are physically close to each other
    bool topology;
    // if true, one sided communication backend will be used
    // otherwise, two sided communication backend is used
    bool one_sided_communication;
    long long memory_limit;
    long long memory_used;
    int n_bfs_steps;
    int n_dfs_steps;

    // constructors
    Strategy();
    // move constructor
    Strategy(Strategy&& other);

    // constructs the Strategy from the command line
    Strategy(const std::string& cmd_line);

    // constructs the Strategy form the command line
    Strategy(int argc, char** argv);

    Strategy(int mm, int nn, int kk, size_t PP, std::vector<int>& divs,
             std::string& dims, std::string& types,
             long long mem_limit = std::numeric_limits<long long>::max(),
             bool top = false);

    Strategy(int mm, int nn, int kk, size_t PP, 
            long long mem_limit = std::numeric_limits<long long>::max(),
            bool top = false);

    // parses the command line options and initializes the varialbes
    void initialize(const std::string& cmd_line);

    // parses steps if defined manually by the user
    void process_steps(size_t start, const std::string& line);

    // greates common divisor of a and b
    int gcd(int a, int b);

    // divides and rounds up long long integers
    static long long divide_and_round_up(long long x, long long y);

    // round to next multiple
    int next_multiple_of(int n_to_round, int multiple);

    // find all divisors of n
    std::vector<int> find_divisors(int n);
    // finds divm, divn and divk such that m/divm = n/divn = k/divk = cubic_root(mnk/P)
    // or at least as close as possible to this such that divm*divn*divk = P
    std::tuple<int, int, int> balanced_divisors(long long m, long long n, long long k, int P);

    // prime decomposition of n
    std::vector<int> decompose(int n);

    // finds divisor of P closest to dimensions/target
    int closest_divisor(int P, int dimension, double target);

    // default strategy dividing always the largest dimension in that step
    // if there is enough memory uses BFS step, if not uses DFS step
    void default_strategy();
    // strategy that tries to make each base case as square as possible
    // it always uses all the resources (all P available ranks) but tries to find
    // divm, divn and divk such that divm * divn * divk = P and m/divm = n/divn = k/divk.
    // if there is not enough memory in some step, then DFS step is performed and new
    // divm, divn and divk are found that correspond to the new subproblem.
    void square_strategy();

    // token is a triplet e.g. bm3 (denoting BFS (m / 3) step)
    void process_token(const std::string& step_triplet);

    void throw_exception(const std::string& message);

    // finds the position after the defined flag or throws an exception 
    // if flag is not found in the line.
    int find_flag(const std::string& short_flag, const std::string& long_flag, 
                     const std::string& message, const std::string& line, 
                     bool throw_exception=true);

    // looks for the defined flag in the line
    // if found return true, otherwise returns false
    bool flag_exists(const std::string& short_flag, const std::string& long_flag, 
            const std::string& line);

    // finds the next int after start in the line
    int next_int(int start, const std::string& line);
    long long next_long_long(int start, const std::string& line);

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

    static long long initial_memory(long long m, long long n, long long k, int P);
    static long long required_memory(Strategy& strategy);

    // checks if the strategy is well-defined
    void check_if_valid();

    friend std::ostream& operator<<(std::ostream& os, const Strategy& other);
};
#endif
