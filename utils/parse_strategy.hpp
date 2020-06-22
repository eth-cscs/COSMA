#include <regex>
#include <vector>

#include <cosma/strategy.hpp>
#include <cosma/environment_variables.hpp>

/*
 * Parses the command line input to get the problem size
 * and divisions strategy if provided in the input.
 * Returns the strategy.
 */

// finds the next int after start in the line
int next_int(int start, const std::string& line) {
    if (start < 0)
        return -1;
    std::regex int_expr("([0-9]+)");
    auto it = std::sregex_iterator(line.begin() + start, line.end(), int_expr);
    int result = std::stoi(it->str());
    return result;
}

// token is a triplet e.g. pm3 (denoting parallel (m / 3) step)
void process_token(const std::string &step_triplet, 
                   std::string& step_type,
                   std::string& split_dimension,
                   std::vector<int>& divisors) {
    if (step_triplet.length() < 3)
        return;
    step_type += step_triplet[0];
    split_dimension += step_triplet[1];
    divisors.push_back(next_int(2, step_triplet));
}

cosma::Strategy parse_strategy(const int m, const int n, 
                               const int k, const int P,
                               const std::vector<std::string>& steps,
                               const long long memory_limit,
                               const bool overlap_comm_and_comp) {
    if (steps.size() == 0) {
        cosma::Strategy strategy(m, n, k, P, memory_limit);
        if (overlap_comm_and_comp) {
            strategy.enable_overlapping_comm_and_comp();
        }
        return strategy;
    } else {
        std::string step_type = "";
        std::string split_dimension = "";
        std::vector<int> divisors;

        for (const std::string& step : steps) {
            process_token(step, step_type, split_dimension, divisors);
        }

        cosma::Strategy strategy(m, n, k, P, divisors, split_dimension, step_type, memory_limit);
        if (overlap_comm_and_comp) {
            strategy.enable_overlapping_comm_and_comp();
        }
        return strategy;
    }
}

