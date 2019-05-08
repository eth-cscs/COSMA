#include <blas.h>
#include <vector>
#include <local_multiply.hpp>
#include <timer.hpp>
#include <chrono>

using namespace cosma;

double sq_score(double a, double b) {
    double result = ((1.0 * a / b) + (1.0 * b / a)) / (2.0 * std::max(1.0 * a/b, 1.0 * b/a));
    // double result = std::min(a, b) / std::max(a, b);
    return result;
}

double score(double m, double n, double k) {
    double score_a = sq_score(m, k);
    double score_b = sq_score(k, n);
    double score_c = sq_score(m, n);
    double result = score_a * score_b * score_c;
    return result;
}

double throughput(double m, double n, double k, double time) {
    return m * n * k * 2 / (1e6 *time);
}

struct problem {
    int m;
    int n;
    int k;

    double time;
    double score;

    double tps;

    problem() = default;
    problem(int mm, int nn, int kk, double tt, double ss, double thr) :
    m(mm), n(nn), k(kk), time(tt), score(ss), tps(thr) {}
};

int main(int argc, char** argv) {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;

    int min_m = 1000;
    int min_n = 1000;
    int min_k = 1000;

    int max_m = 50000;
    int max_n = 1000;
    int max_k = 1000;


    int step_m = 500;
    int step_n = 500;
    int step_k = 500;

    int n_rep = 2;

    // run random dgemm in order to initialize it
    for (int i = 0; i < n_rep; ++i) {
        a = std::vector<double>(min_m*min_m);
        b = std::vector<double>(min_m*min_m);
        c = std::vector<double>(min_m*min_m);

        local_multiply_cpu(a.data(), b.data(), c.data(), min_m, min_m, min_m, 0.0);
    }

    std::vector<problem> timings;

    for (int m = min_m; m <= max_m; m += step_m) {
        for (int n = min_n; n <= max_n; n += step_n) {
            for (int k = min_k; k <= max_k; k += step_k) {
                auto start = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < n_rep; ++rep) {
                    a = std::vector<double>(m * k);
                    b = std::vector<double>(k * n);
                    c = std::vector<double>(m * n);

                    local_multiply_cpu(a.data(), b.data(), c.data(), m, n, k, 0.0);
                }
                auto finish = std::chrono::high_resolution_clock::now();
                auto time = std::chrono::duration_cast<std::chrono::milliseconds>
                (finish - start).count();
                time /= 1.0 * n_rep;
                double mul_score = score(m, n, k);
                double tps = throughput(m, n, k, time);
                problem prob(m, n, k, time, mul_score, tps);
                timings.push_back(prob);
            }
        }
    }

    std::sort(timings.begin(), timings.end(), [](const problem& lhs, const problem& rhs) {
        return lhs.tps < rhs.tps;
    });

    for (auto& problem : timings) {
        std::cout << problem.m << " " << problem.tps << " " << problem.score << std::endl;
        // std::cout << "(" << problem.m << ", " << problem.n << ", " << problem.k << "), tps = " << problem.tps << ", score = " << problem.score << std::endl;
    }
    return 0;
}
