#include <cosma/math_utils.hpp>

namespace cosma {
int math_utils::gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

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
math_utils::balanced_divisors(long long m, long long n, long long k,
                              int P, int min_local_problem_size) {
    // each divisor can be at most the value of the dimension
    auto max_divm = std::min(
                            // layout condition + mathematical
                            // the matrix that is not split here (i.e. B)
                            // must have #colums >= divm
                            std::min(m, n), 
                            // min_problem_size condition
                            m/min_local_problem_size); // min_prob_size condition
    max_divm = std::max(1LL, max_divm);
    auto max_divn = std::min(std::min(k, n),
                             n/min_local_problem_size);
    max_divn = std::max(1LL, max_divn);
    auto max_divk = std::min(std::min(k, n),
                             k/min_local_problem_size);
    max_divk = std::max(1LL, max_divk);

    // protect from overflow by adding redundant checks
    if (max_divm < P && max_divn < P && max_divk < P
            && max_divm * max_divn < P
            && max_divm * max_divn * max_divk < P) {
        P = (int) (max_divm * max_divn * max_divk);
    }

    // sort the dimensions
    std::vector<int> dims = {(int)m, (int)n, (int)k};
    std::sort(dims.begin(), dims.end());

    double target_tile_size = 0.0;
    // avoid overflow
    if (dims[2] >= P) {
        target_tile_size = std::cbrt(1.0 * dims[2]/ P * dims[0] * dims[1]);
    } else if (dims[1] * dims[2] >= P) {
        target_tile_size = std::cbrt(1.0 * dims[1] * dims[2] / P * dims[0]);
    } else {
        target_tile_size = std::cbrt(1.0 * dims[0] * dims[1] * dims[2] / P);
    }

    int error = std::numeric_limits<int>::max();
    int divm = 1;
    int divn = 1;
    int divk = 1;

    for (const int &div1 : find_divisors(P)) {
        if (div1 > max_divm) break;

        int error_lower_bound = std::abs(m / div1 - target_tile_size);
        if (error_lower_bound > error) {
            continue;
        }
        for (const int &div2 : find_divisors(P / div1)) {
            if (div2 > max_divn) break;
            int div3 = std::min((P / div1) / div2, (int) max_divk);
            int current_error = std::abs(m / div1 - target_tile_size) +
                                std::abs(n / div2 - target_tile_size) +
                                std::abs(k / div3 - target_tile_size);
            // prefer new divisors if they make tile size closer to the target size
            // or if they utilize more processors
            if (div1 * div2 * div3 > divm * divn * divk ||
                div1 * div2 * div3 == divm * divn * divk && current_error < error) {
                divm = div1;
                divn = div2;
                divk = div3;

                error = current_error;
            }
        }
    }
    return std::make_tuple(divm, divn, divk);
}

// find all prime factors of a given number n
std::vector<int> math_utils::decompose(int n) {
    std::vector<int> factors;
    int orig_n = n;

    // number of 2s that divide n
    while (n % 2 == 0) {
        factors.push_back(2);
        n = n / 2;
    }

    // n must be odd at this point.
    // we can skip one element
    for (int i = 3; i <= std::sqrt(n); i = i + 2) {
        // while i divides n, print i and divide n
        while (n % i == 0) {
            factors.push_back(i);
            n = n / i;
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
        error = std::abs(1.0 * dimension / i - target);

        if (error <= best_error) {
            best_div = i;
            best_error = error;
        }
    }

    return best_div;
}

int math_utils::int_div_up(int numerator, int denominator) {
    return numerator / denominator +
           (((numerator < 0) ^ (denominator > 0)) && (numerator % denominator));
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
    int w = (int)std::floor((std::sqrt(8 * z + 1) - 1) / 2);
    int t = (w * w + w) / 2;
    int y = z - t;
    int x = w - y;
    return {x, y};
}

// maps (N, N) -> N
int math_utils::cantor_pairing(const int i, const int j) {
    int sum = i + j;
    return (sum * (sum + 1)) / 2 + j;
}
} // namespace cosma
