#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace cosma {
namespace math_utils {
// greates common divisor of a and b
int gcd(int a, int b);

// divides and rounds up long long integers
long long divide_and_round_up(long long x, long long y);

// round to next multiple
int next_multiple_of(int n_to_round, int multiple);

// check if the number is a power of 2
bool is_power_of_2(std::size_t n);

// find the next power of 2 that is > than n
std::size_t next_greater_power_of_2(std::size_t n, std::size_t power_of_2 = 1);

// find the next power of 2 that is >= n
std::size_t next_power_of_2(std::size_t n);

// find all divisors of n
std::vector<int> find_divisors(int n);
// Finds the divisors dm, dn and dk for m, n and k respectively, such that
// 1. dm * dn * dk <= P
// 2. dm <= min(m, n, m/local_problem_size)
// 3. dn <= min(n, k, n/local_problem_size)
// 5. dk <= min(k, n, k/local_problem_size)
// 6. balanced: m/dm approx= n/dn approx= k/dk
//
// For the upper bound on divisors, the following conditions are taken into account:
//     - layout-conditions: the matrix that is not split, i.e. which does not
//                          contain the split dimension, must have #columns
//                          at least as large as the divisor of that dimension
//     - min-problem-size: the minimum size of the corresponding dimension
//                         after splitting should be at least min_problem_size
//     - mathematical: divisor or some dimension should be at least 1 (i.e.
// 
std::tuple<int, int, int>
balanced_divisors(long long m, long long n, long long k, 
                          int P, int min_problem_size);

// prime decomposition of n
std::vector<int> decompose(int n);

// finds divisor of P closest to dimensions/target
int closest_divisor(int P, int dimension, double target);

// divide numerator by denominator and round it up to int
int int_div_up(int numerator, int denominator);

// returns a value (0, 1], that describes how close to the square matrix,
// the matrix with dimensions rows x cols is.
double square_score(int rows, int cols);

// returns a value (0, 1] that describes how close the performance
// of gemm(m, n, k) is to the performance of a corresponding square case
// gemm(q, q, q) where q = cubic_root(m*n*k)
double square_score(int m, int n, int k);

int cantor_pairing(const int i, const int j);
std::pair<int, int> invert_cantor_pairing(int z);
}; // namespace math_utils
} // namespace cosma
