#include <cosma/strategy.hpp>
#include <cosma/environment_variables.hpp>
#include <vector>

/*
 * Parses the command line input to get the problem size
 * and divisions strategy if provided in the input.
 * Returns the strategy.
 */

// token is a triplet e.g. pm3 (denoting parallel (m / 3) step)
void process_token(const std::string &step_triplet, 
                   std::string& step_type,
                   std::string& split_dimension,
                   std::vector<int>& divisors) {
    if (step_triplet.length() < 3)
        return;
    step_type += step_triplet[0];
    split_dimension += step_triplet[1];
    divisors.push_back(options::next_int(2, step_triplet));
}

cosma::Strategy parse_strategy(int m, int n, int k,
                               std::vector<std::string>& steps,
                               long long memory_limit,
                               bool overlap_comm_and_comp) {
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

