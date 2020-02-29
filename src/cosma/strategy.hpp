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
    int m = 0;
    int n = 0;
    int k = 0;
    // number of processors
    size_t P = 0;

    long long memory_limit = 0;

    // minimum problem size per rank
    // the total number of ranks will be reduced
    // if the problem size per rank is too small
    // by default = 32
    static int min_dim_size;

    // the actual minimum problem size
    // that is induced by given strategy
    int min_m = 0;
    int min_n = 0;
    int min_k = 0;

    // beta parameter of gemm
    double beta = 0.0;
    // stores the divisor in each step of the algorithm
    std::vector<int> divisors = {};
    // returns m, n or k character depending on
    // which dimension was split in each step
    std::string split_dimension = "";
    // describes whether a sequential step (s) or a parallel step (p) is used in
    // each step
    std::string step_type = "";
    // if true, MPI will try to relabel ranks such that
    // the ranks which communicate are physically close to each other
    bool topology = false;
    // if true, uses busy waiting in the thread performing MPI communication
    // otherwise, uses polling to query if the communication request has
    // completed
    bool use_busy_waiting = true;
    long long memory_used = 0;
    int n_parallel_steps = 0;
    int n_sequential_steps = 0;

    int n_parallel_steps_before_gemm_a = 0;
    int n_parallel_steps_before_gemm_b = 0;
    int n_parallel_steps_before_gemm_c = 0;

    static void disable_optimization();

    // constructors
    Strategy();
    // copy constructor
    Strategy(Strategy &other);
    Strategy(const Strategy &other);

    // Strategy& operator=(const Strategy& other) = default;
    // Strategy& operator=(Strategy& other) = default;

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
             bool overlap = false,
             bool busy_waiting = true);

    Strategy(int mm,
             int nn,
             int kk,
             size_t PP,
             long long mem_limit = std::numeric_limits<long long>::max(),
             double b = 0.0,
             bool top = false,
             bool overlap = false,
             bool busy_waiting = true);

    // number of steps of the algorithm
    size_t n_steps() const;

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
    void square_strategy(bool& should_optimize);

    void spartition_strategy();

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

    // reduces the number of processors if problem size
    // gets smaller than min_dim_size
    void optimize_strategy();
    // checks if the strategy is well-defined
    void check_if_valid();
    void check_if_overlap_possible();
    // prefers a single division by (a*b) over two divisions (one by a and one
    // by b)
    void compress_steps();

    bool should_overlap_comm_and_comp(int step) const;

    bool operator==(const Strategy &other) const;
    bool operator!=(const Strategy &other) const;

    friend std::ostream &operator<<(std::ostream &os, const Strategy &other);

    void compute_min_sizes();

    // if number of processes is 0, then n_steps = 0
    // and then the strategy is considered empty
    bool empty() const;

    // returns dimensions of a matrix with given label
    // where label = A, B or C
    int n_rows(char label) const;
    int n_cols(char label) const;

    void enable_overlapping_comm_and_comp();

    void check_if_irregular();

    // the strategy is considered irregular if any dimension
    // (at any step) is divided by a divisor that does not perfectly
    // divide that dimension
    bool irregular = true;

  private:
    // if true, the communication and computation will be overlapped
    // this variable should not be changed outside of the class but only
    // through the function `enable_overlapping_comm_and_comp`.
    // because this function also has to update the variable `irregular`
    // when the overlap is turned on.
    bool overlap_comm_and_comp = false;

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
