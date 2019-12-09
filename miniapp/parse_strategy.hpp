#include <cosma/strategy.hpp>
#include <options.hpp>
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

void process_steps(size_t start, const std::string &line,
                   std::string& step_type,
                   std::string& split_dimension,
                   std::vector<int>& divisors) {
    // go to the end of the string or space
    auto end = line.find(start, ' ');
    if (end == std::string::npos) {
        end = line.length();
    }
    std::string substring = line.substr(start, end);
    std::istringstream stream(substring);
    std::string token;

    while (std::getline(stream, token, ',')) {
        process_token(token, step_type, split_dimension, divisors);
    }
}

cosma::Strategy parse_strategy(int argc, char** argv) {
    std::string cmd_line;
    for (auto i = 1; i < argc; ++i) {
        cmd_line += std::string(argv[i]) + " ";
    }
    int m_it = options::find_flag(
        "-m", "--m_dimension", "Dimension m has to be defined.", cmd_line);
    int n_it = options::find_flag(
        "-n", "--n_dimension", "Dimension n has to be defined.", cmd_line);
    int k_it = options::find_flag(
        "-k", "--k_dimension", "Dimension k has to be defined.", cmd_line);
    int P_it = options::find_flag("-P",
                                   "--processors",
                                   "Number of processors has to be defined.",
                                   cmd_line);
    int M_it = options::find_flag(
        "-L",
        "--memory",
        "Memory limit: maximum number of elements each rank can own.",
        cmd_line,
        false);
    int m = options::next_int(m_it, cmd_line);
    int n = options::next_int(n_it, cmd_line);
    int k = options::next_int(k_it, cmd_line);
    int P = options::next_int(P_it, cmd_line);
    long long memory_limit = options::next_long_long(M_it, cmd_line);

    // if memory limit not given, assume we have infinity
    // (i.e. assume that each rank can store all 3 matrices)
    if (memory_limit < 0) {
        memory_limit = std::numeric_limits<long long>::max();
    }

    bool overlap_comm_and_comp = options::flag_exists("-o", "--overlap", cmd_line);

    bool steps_predefined = options::flag_exists("-s", "--steps", cmd_line);

    if (steps_predefined) {
        auto steps_it = options::find_flag(
            "-s", "--steps", "Division steps have to be defined.", cmd_line);
        std::string step_type = "";
        std::string split_dimension = "";
        std::vector<int> divisors;
        process_steps(steps_it, cmd_line, step_type, split_dimension, divisors);
        cosma::Strategy strategy(m, n, k, P, divisors, split_dimension, step_type, memory_limit);
        strategy.overlap_comm_and_comp = overlap_comm_and_comp;
        return strategy;
    } else {
        cosma::Strategy strategy(m, n, k, P, memory_limit);
        strategy.overlap_comm_and_comp = overlap_comm_and_comp;
        return strategy;
    }
}

