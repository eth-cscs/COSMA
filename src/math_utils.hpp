#pragma once
#include <math.h>
#include <limits>
#include <vector>
#include <tuple>
#include <algorithm>

namespace math_utils {
    // greates common divisor of a and b
    int gcd(int a, int b);

    // divides and rounds up long long integers
    long long divide_and_round_up(long long x, long long y);

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
};
