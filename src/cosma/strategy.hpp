#pragma once

#include <cosma/math_utils.hpp>

#include <iostream>
#include <limits>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace cosma {
class Strategy {
  public:
    // matrix dimensions
    int m;
    int n;
    int k;

    int min_m;
    int min_n;
    int min_k;

    // number of processors
    size_t P;
    long long memory_limit;
    // beta parameter of gemm
    double beta = 0.0;
    // stores the divisor in each step of the algorithm
    std::vector<int> divisors;
    // returns m, n or k character depending on
    // which dimension was split in each step
    std::string split_dimension = "";
    // describes whether a sequential step (s) or a parallel step (p) is used in
    // each step
    std::string step_type = "";
    // number of steps of the algorithm
    size_t n_steps = 0;
    // if true, MPI will try to relabel ranks such that
    // the ranks which communicate are physically close to each other
    bool topology;
    // if true, the communication and computation will be overlapped
    bool overlap_comm_and_comp = true;
    // if true, uses busy waiting in the thread performing MPI communication
    // otherwise, uses polling to query if the communication request has
    // completed
    bool use_busy_waiting = true;
    long long memory_used;
    int n_parallel_steps = 0;
    int n_sequential_steps = 0;

    int n_parallel_steps_before_gemm_a;
    int n_parallel_steps_before_gemm_b;
    int n_parallel_steps_before_gemm_c;

    // constructors
    Strategy();
    // move constructor
    Strategy(Strategy &&other);

    // constructs the Strategy from the command line
    Strategy(const std::string &cmd_line);

    // constructs the Strategy form the command line
    Strategy(int argc, char **argv);

    Strategy(int mm,
             int nn,
             int kk,
             size_t PP,
             std::vector<int> &divs,
             std::string &dims,
             std::string &types,
             long long mem_limit = std::numeric_limits<long long>::max(),
             double b = 0.0,
             bool top = false,
             bool overlap = true,
             bool busy_waiting = true);

    Strategy(int mm,
             int nn,
             int kk,
             size_t PP,
             std::string steps,
             long long mem_limit = std::numeric_limits<long long>::max(),
             double b = 0.0,
             bool top = false,
             bool overlap = true,
             bool busy_waiting = true);

    Strategy(int mm,
             int nn,
             int kk,
             size_t PP,
             long long mem_limit = std::numeric_limits<long long>::max(),
             double b = 0.0,
             bool top = false,
             bool overlap = true,
             bool busy_waiting = true);

    // parses the command line options and initializes the varialbes
    void initialize(const std::string &cmd_line);

    // parses steps if defined manually by the user
    void process_steps(size_t start, const std::string &line);

    // default strategy dividing always the largest dimension in that step
    // if there is enough memory uses a parallel step, if not uses a sequential
    // step
    void default_strategy();
    // strategy that tries to make each base case as square as possible
    // it always uses all the resources (all P available ranks) but tries to
    // find divm, divn and divk such that divm * divn * divk = P and m/divm =
    // n/divn = k/divk. if there is not enough memory in some step, then a
    // sequential step is performed and new divm, divn and divk are found that
    // correspond to the new subproblem.
    void square_strategy();

    void spartition_strategy();

    // token is a triplet e.g. pm3 (denoting parallel (m / 3) step)
    void process_token(const std::string &step_triplet);

    void throw_exception(const std::string &message);

    bool split_m(size_t i) const;
    bool split_n(size_t i) const;
    bool split_k(size_t i) const;

    bool split_A(size_t i) const;
    bool split_B(size_t i) const;
    bool split_C(size_t i) const;
    bool split(char label, size_t i) const;

    bool sequential_step(size_t i) const;
    bool parallel_step(size_t i) const;

    int divisor(size_t i) const;
    int divisor_m(size_t i) const;
    int divisor_n(size_t i) const;
    int divisor_k(size_t i) const;

    int divisor_row(char matrix, size_t i) const;
    int divisor_col(char matrix, size_t i) const;

    bool final_step(size_t i) const;
    int parallel_steps_before_gemm(char label) const;

    static long long
    initial_memory(long long m, long long n, long long k, int P);
    static long long required_memory(Strategy &strategy);

    // checks if the strategy is well-defined
    void check_if_valid();
    void check_if_overlap_possible();
    // prefers a single division by (a*b) over two divisions (one by a and one
    // by b)
    void compress_steps();

    bool should_overlap_comm_and_comp(int step) const;

    friend std::ostream &operator<<(std::ostream &os, const Strategy &other);

    void compute_min_sizes();

  private:
    bool divide(std::vector<int> &div_factors,
                int &dim_i,
                long long &dim1,
                long long &dim2,
                long long &dim3,
                int &P,
                long long &needed_memory,
                const std::string label);
};
} // namespace cosma
