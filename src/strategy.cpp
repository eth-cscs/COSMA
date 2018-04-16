#include "strategy.hpp"

// constructors
Strategy::Strategy() = default;
// move constructor
Strategy::Strategy(Strategy&& other) = default;

// constructs the Strategy from the command line
Strategy::Strategy(const std::string& cmd_line) {
    initialize(cmd_line);
}

// constructs the Strategy form the command line
Strategy::Strategy(int argc, char** argv) {
    std::string input;
    for (auto i = 1; i < argc; ++i) {
        input += argv[i];
    }
    initialize(input);
}

Strategy::Strategy(int mm, int nn, int kk, size_t PP, std::vector<int>& divs,
        std::string& dims, std::string& types, bool top) : m(mm), n(nn), k(kk), P(PP),
        topology(top) {
    divisors = divs;
    split_dimension = dims;
    step_type = types;
    n_steps = divisors.size();
}

void Strategy::initialize(const std::string& cmd_line) {
    auto m_it = find_flag("-m", "--m_dimension", "Dimension m has to be defined.", cmd_line);
    auto n_it = find_flag("-n", "--n_dimension", "Dimension n has to be defined.", cmd_line);
    auto k_it = find_flag("-k", "--k_dimension", "Dimension k has to be defined.", cmd_line);
    auto P_it = find_flag("-P", "--processors", "Number of processors has to be defined.", cmd_line);
    topology = find_bool_flag("-t", "--topology", "If true, MPI topology will be adapted\
            to the communication.", cmd_line);
    auto steps_it = find_flag("-s", "--steps", "Division steps have to be defined.", cmd_line);
    m = next_int(m_it, cmd_line);
    n = next_int(n_it, cmd_line);
    k = next_int(k_it, cmd_line);
    P = next_int(P_it, cmd_line);

    process_steps(steps_it, cmd_line);
    n_steps = divisors.size();

    check_if_valid();
}

void Strategy::process_steps(size_t start, const std::string& line) {
    // go to the end of the string or space
    auto end = line.find(start, ' ');
    if (end == std::string::npos) {
        end = line.length();
    }
    std::string substring = line.substr(start, end);
    std::istringstream stream(substring);
    std::string token;

    while (std::getline(stream, token, ',')) {
        process_token(token);
    }
}

// token is a triplet e.g. bm3 (denoting BFS (m / 3) step
void Strategy::process_token(const std::string& step_triplet) {
    if (step_triplet.length() < 3) return;
    step_type += step_triplet[0];
    split_dimension += step_triplet[1];
    divisors.push_back(next_int(2, step_triplet));
}

void Strategy::throw_exception(const std::string& message) {
    std::cout << "Splitting strategy not well defined.\n";
    throw std::runtime_error(message);
}

// finds the position after the defined flag or throws an exception 
// if flag is not found in the line.
size_t Strategy::find_flag(const std::string& short_flag, const std::string& long_flag, 
        const std::string& message, const std::string& line) {
    auto position = line.find(short_flag);
    auto length = short_flag.length();
    if (position == std::string::npos) {
        position = line.find(long_flag);
        length = long_flag.length();
        if (position == std::string::npos) {
            throw_exception(message);
        }
    }
    while (line[position + length] == ' ') length++;
    return position + length;
}

// looks for the defined flag in the line
// if found return true, otherwise returns false
bool Strategy::find_bool_flag(const std::string& short_flag, const std::string& long_flag, 
        const std::string& message, const std::string& line) {
    auto position = line.find(short_flag);
    if (position == std::string::npos) {
        position = line.find(long_flag);
        if (position == std::string::npos) {
            return false;
        }
    }
    return true;
}

// finds the next int after start in the line
int Strategy::next_int(size_t start, const std::string& line) {
    std::regex int_expr("([0-9]+)");
    auto it = std::sregex_iterator(line.begin() + start, line.end(), int_expr);
    int result = std::stoi(it->str());
    return result;
}

const bool Strategy::split_m(size_t i) const {
    return split_dimension[i] == 'm';
}

const bool Strategy::split_n(size_t i) const {
    return split_dimension[i] == 'n';
}

const bool Strategy::split_k(size_t i) const {
    return split_dimension[i] == 'k';
}

const bool Strategy::split_A(size_t i) const {
    return split_m(i) || split_k(i);
}

const bool Strategy::split_B(size_t i) const {
    return split_k(i) || split_n(i);
}

const bool Strategy::split_C(size_t i) const {
    return split_m(i) || split_n(i);
}

const bool Strategy::dfs_step(size_t i) const {
    return step_type[i] == 'd';
}

const bool Strategy::bfs_step(size_t i) const {
    return step_type[i] == 'b';
}

const int Strategy::divisor(size_t i) const {
    return divisors[i];
}

const int Strategy::divisor_m(size_t i) const {
    return split_m(i) ? divisors[i] : 1;
}

const int Strategy::divisor_n(size_t i) const {
    return split_n(i) ? divisors[i] : 1;
}

const int Strategy::divisor_k(size_t i) const {
    return split_k(i) ? divisors[i] : 1;
}

const int Strategy::divisor_row(char matrix, size_t i) const {
    if (matrix == 'A')
        return divisor_m(i);
    if (matrix == 'B')
        return divisor_k(i);
    if (matrix == 'C')
        return divisor_m(i);
    return 1;
}

const int Strategy::divisor_col(char matrix, size_t i) const {
    if (matrix == 'A')
        return divisor_k(i);
    if (matrix == 'B')
        return divisor_n(i);
    if (matrix == 'C')
        return divisor_n(i);
    return 1;
}

const bool Strategy::final_step(size_t i) const {
    return i == n_steps;
}

// checks if the strategy is well-defined
void Strategy::check_if_valid() {
    int mi = m;
    int ni = n;
    int ki = k;
    int Pi = P;

    for (size_t i = 0; i < n_steps; ++i) {
        if (divisors[i] <= 1) {
            throw_exception("Divisors in each step must be larger than 1. \
                    Divisor in step " + std::to_string(i) + " = " 
                    + std::to_string(divisors[i]) + ".");
        }

        if (split_dimension[i] != 'm' && split_dimension[i] != 'n' 
                && split_dimension[i] != 'k') {
            throw_exception("Split dimension in each step must be m, n or k");
        }

        if (step_type[i] != 'b' && step_type[i] != 'd') {
            throw_exception("Step type should be either b or d.");
        }

        if (step_type[i] == 'b') {
            if (Pi <= 1) {
                throw_exception("Not enough processors for this division strategy \
                        The product of all divisors in a BFS step should be equal \
                        to the number of processors");
            }

            if (Pi % divisors[i] != 0) {
                throw_exception("The number of processors left in each BFS step \
                        should be divisible by divisor.");
            }

            Pi /= divisors[i];
        }

        if (split_dimension[i] == 'm') {
            mi /= divisors[i];
        }

        // since we are using column major ordering, the #columns of each matrix must be at least
        // the number of processors left at that step
        if (split_dimension[i] == 'n') {
            ni /= divisors[i];
            if (ni < Pi) {
                throw_exception("Dimension n at step " + std::to_string(i) + " = "
                        + std::to_string(ni) + ", which is less than the number of processors left = "
                        + std::to_string(Pi));
            }
        }

        if (split_dimension[i] == 'k') {
            ki /= divisors[i];
            if (ki < Pi) {
                throw_exception("Dimension k at step " + std::to_string(i) + " = "
                        + std::to_string(ki) + ", which is less than the number  \
                        of processors left = "
                        + std::to_string(Pi));
            }
        }
    }
    if (Pi != 1) {
        throw_exception("Too many processors. The number of processors should be \
                equal to the product of divisors in all BFS steps.");
    }
}

std::ostream& operator<<(std::ostream& os, const Strategy& other) {
    os << "Matrix dimensions (m, n, k) = (" << other.m << ", " << other.n << ", " 
        << other.k << ")\n";
    os << "Number of processors: " << other.P << "\n";
    os << "Divisions strategy: \n";
    for (size_t i = 0; i < other.n_steps; ++i) {
        if (other.step_type[i] == 'b') {
            os << "BFS (" << other.split_dimension[i] << " / " << other.divisors[i] << ")\n";
        } else {
            os << "DFS (" << other.split_dimension[i] << " / " << other.divisors[i] << ")\n";
        }
    }
    return os;
}
