#include "math_utils.hpp"

int math_utils::gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

long long math_utils::divide_and_round_up(long long x, long long y) {
    return 1 + ((x - 1) / y);
}

int math_utils::next_multiple_of(int n_to_round, int multiple) {
    if (multiple == 0)
        return n_to_round;

    int remainder = n_to_round % multiple;
    if (remainder == 0)
        return n_to_round;

    return n_to_round + multiple - remainder;
}

// find all divisors of a given number n
std::vector<int> math_utils::find_divisors(int n) {
    std::vector<int> divs;
    for (int i = 1; i < n; ++i) {
        if (n % i == 0) {
            divs.push_back(i);
        }
    }
    return divs;
}

std::tuple<int, int, int> math_utils::balanced_divisors(long long m, long long n, long long k, int P) {
    // sort the dimensions 
    std::vector<long long> dimensions = {m, n, k};
    std::sort(dimensions.begin(), dimensions.end());

    // find divm, divn, divk such that m/divm = n/divn = k/divk (as close as possible)
    // be careful when dividing, since the product mnk can be very large
    double target_tile_size = std::cbrt(1.0*dimensions[1]*dimensions[2] / P * dimensions[0]);
    int divk = closest_divisor(P, k, target_tile_size);
    P /= divk;
    int divn = closest_divisor(P, n, target_tile_size);
    P /= divn;
    int divm = P;

    return std::make_tuple(divm, divn, divk);
}

// find all prime factors of a given number n
std::vector<int> math_utils::decompose(int n) {
    std::vector<int> factors;

    // number of 2s that divide n
    while (n%2 == 0) {
        factors.push_back(2);
        n = n/2;
    }

    // n must be odd at this point. 
    // we can skip one element
    for (int i = 3; i <= std::sqrt(n); i = i+2) {
        // while i divides n, print i and divide n
        while (n%i == 0) {
            factors.push_back(i);
            n = n/i;
        }
    }

    // This condition is to handle the case when n
    // is a prime number greater than 2
    if (n > 2) {
        factors.push_back(n);
    }
    return factors;
}

int math_utils::closest_divisor(int P, int dimension, double target) {
    int divisor = 1;
    int error;
    int best_error = std::numeric_limits<int>::max();
    int best_div = 1;

    for (int i : find_divisors(P)) {
        error = std::abs(1.0*dimension / i - target);

        if (error < best_error) {
            best_div = i;
            best_error = error;
        }
    }

    return best_div;
}
