#include "math_utils.hpp"

namespace cosma {
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
    for (int i = 1; i <= n; ++i) {
        if (n % i == 0) {
            divs.push_back(i);
        }
    }
    return divs;
}

std::tuple<int, int, int> math_utils::balanced_divisors(long long m, long long n, long long k, int P) {
    // sort the dimensions 
    std::vector<int> dimensions = {(int)m, (int)n, (int)k};
    std::sort(dimensions.begin(), dimensions.end());

    double target_tile_size = std::cbrt(1.0*dimensions[1]*dimensions[2] / P * dimensions[0]);
    // std::cout << "target size = " << target_tile_size << std::endl;

    int error = std::numeric_limits<int>::max();
    int divm = 1;
    int divn = 1;
    int divk = 1;

    for (const int& div1 : find_divisors(P)) {
        int error_lower_bound = std::abs(m/div1 - target_tile_size);
        if (error_lower_bound > error) {
            // std::cout << "skipping " << error_lower_bound << std::endl;
            continue;
        }
        for (const int& div2 : find_divisors(P/div1)) {
            int div3 = (P / div1) / div2;
            // std::cout << "div1 = " << div1 << ", div2 = " << div2 << ", div3 = " << div3 << std::endl;

            int current_error = std::abs(m/div1 - target_tile_size)
                              + std::abs(n/div2 - target_tile_size)
                              + std::abs(k/div3 - target_tile_size);

            // std::cout << "error = " << current_error << std::endl;

            if (current_error < error) {
                divm = div1;
                divn = div2;
                divk = div3;

                error = current_error;
            }

        }
    }
    // std::cout << "balanced divisors of " << m << ", " << n << ", " << k << " are " << divm << ", " << divn << ", " << divk << std::endl;
    return std::make_tuple(divm, divn, divk);
}

/*
std::tuple<int, int, int> math_utils::balanced_divisors(long long m, long long n, long long k, int P) {
    // sort the dimensions 
    std::vector<int> dimensions = {(int)m, (int)n, (int)k};
    std::sort(dimensions.begin(), dimensions.end());

    int orig_P = P;

    // find divm, divn, divk such that m/divm = n/divn = k/divk (as close as possible)
    // be careful when dividing, since the product mnk can be very large
    double target_tile_size = std::cbrt(1.0*dimensions[1]*dimensions[2] / P * dimensions[0]);
    // std::cout << "target tile dimension = " << target_tile_size << std::endl;
    //
    std::cout << "target_size = " << target_tile_size << std::endl;

    int div2 = closest_divisor(P, dimensions[2], target_tile_size);
    P /= div2;

    target_tile_size = std::sqrt(1.0*dimensions[0]*dimensions[1] / P);

    std::cout << "target_size = " << target_tile_size << std::endl;

    int div1 = closest_divisor(P, dimensions[1], target_tile_size);
    P /= div1;
    int div0 = P;

    std::cout << "div0 = " << div0 << ", div1 = " << div1 << ", div2 = " << div2 << std::endl;

    std::vector<int> divisors = {div0, div1, div2};

    auto m_iter = std::find(dimensions.begin(), dimensions.end(), m);
    int m_diff = m_iter - dimensions.begin();
    int divm = divisors[m_diff];
    dimensions.erase(m_iter);
    divisors.erase(divisors.begin() + m_diff);

    auto n_iter = std::find(dimensions.begin(), dimensions.end(), n);
    int n_diff = n_iter - dimensions.begin();
    int divn = divisors[n_diff];
    dimensions.erase(n_iter);
    divisors.erase(divisors.begin() + n_diff);

    int divk = divisors[0];

    std::cout << "balanced divisors of " << m << ", " << n << ", " << k << ", " << orig_P << " are " << divm << ", " << divn << ", " << divk << std::endl;
    return std::make_tuple(divm, divn, divk);
}
*/

// find all prime factors of a given number n
std::vector<int> math_utils::decompose(int n) {
    std::vector<int> factors;
    int orig_n = n;

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

    // std::cout << "factors of " << orig_n << " are: ";
    // for (const auto& el : factors)
    //     std::cout << el << ", ";
    // std::cout << std::endl;
    return factors;
}

int math_utils::closest_divisor(int P, int dimension, double target) {
    int divisor = 1;
    int error;
    int best_error = std::numeric_limits<int>::max();
    int best_div = 1;

    for (int i : find_divisors(P)) {
        error = std::abs(1.0*dimension / i - target);

        if (error <= best_error) {
            best_div = i;
            best_error = error;
        }
    }

    std::cout << "closest divisor of " << dimension << " is " << best_div << std::endl;
    return best_div;
}

int math_utils::int_div_up(int numerator, int denominator) {
    return numerator / denominator +
        (((numerator < 0) ^ (denominator > 0)) && (numerator%denominator));
}

double math_utils::square_score(int rows, int cols) {
    if (rows == 0 || cols == 0) {
        std::runtime_error("square_score function called with zero-dimension.");
    }
    double ratio1 = 1.0 * rows / cols;
    double ratio2 = 1.0 * cols / rows;
    return (ratio1 + ratio2) / (2.0 * std::max(ratio1, ratio2));
}

double math_utils::square_score(int m, int n, int k) {
    double score_a = square_score(m, k);
    double score_b = square_score(k, n);
    double score_c = square_score(m, n);

    return score_a * score_b * score_c;
}

std::pair<int, int> math_utils::invert_cantor_pairing(int z) {
    int w = (int) std::floor((std::sqrt(8 * z + 1) - 1)/2);
    int t = (w * w + w) / 2;
    int y = z - t;
    int x = w - y;
    return {x, y};
}

// maps (N, N) -> N
int math_utils::cantor_pairing(const int i, const int j) {
    int sum = i + j;
    return (sum * (sum + 1))/2 + j;
}
}
