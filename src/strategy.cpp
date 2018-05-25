#include "strategy.hpp"

// constructors
Strategy::Strategy() = default;
// move constructor
Strategy::Strategy(Strategy&& other) = default;

// constructs the Strategy from the command line
Strategy::Strategy(const std::string& cmd_line) {
    initialize(cmd_line);
    n_steps = divisors.size();
    check_if_valid();
}

// constructs the Strategy form the command line
Strategy::Strategy(int argc, char** argv) {
    std::string input;
    for (auto i = 1; i < argc; ++i) {
        input += argv[i];
    }
    initialize(input);
    n_steps = divisors.size();
    check_if_valid();
}

Strategy::Strategy(int mm, int nn, int kk, size_t PP, std::vector<int>& divs,
        std::string& dims, std::string& types, long long mem_limit, bool top) : m(mm), n(nn), k(kk), P(PP), memory_limit(mem_limit), topology(top) {
    divisors = divs;
    split_dimension = dims;
    step_type = types;
    n_steps = divisors.size();
    check_if_valid();
}

Strategy::Strategy(int mm, int nn, int kk, size_t PP, long long mem_limit, bool top) : 
    m(mm), n(nn), k(kk), P(PP), memory_limit(mem_limit), topology(top) {
    square_strategy();
    //default_strategy();
    // spartition_strategy();
    n_steps = divisors.size();
    check_if_valid();
}

void Strategy::initialize(const std::string& cmd_line) {
    auto m_it = find_flag("-m", "--m_dimension", "Dimension m has to be defined.", cmd_line);
    auto n_it = find_flag("-n", "--n_dimension", "Dimension n has to be defined.", cmd_line);
    auto k_it = find_flag("-k", "--k_dimension", "Dimension k has to be defined.", cmd_line);
    auto P_it = find_flag("-P", "--processors", 
            "Number of processors has to be defined.", cmd_line);
    auto M_it = find_flag("-L", "--memory", 
            "Memory limit: maximum number of elements each rank can own.", cmd_line, false);
    m = next_int(m_it, cmd_line);
    n = next_int(n_it, cmd_line);
    k = next_int(k_it, cmd_line);
    P = next_int(P_it, cmd_line);
    memory_limit = next_long_long(M_it, cmd_line);

    // if memory limit not given, assume we have infinity
    // (i.e. assume that each rank can store all 3 matrices)
    if (memory_limit < 0) {
        memory_limit = std::numeric_limits<long long>::max();
    }

    topology = flag_exists("-t", "--topology", cmd_line);

    bool steps_predefined = flag_exists("-s", "--steps", cmd_line);

    if (steps_predefined) {
        auto steps_it = find_flag("-s", "--steps", "Division steps have to be defined.", cmd_line);
        process_steps(steps_it, cmd_line);
    }
    else {
        square_strategy();
        //default_strategy();
        //spartition_strategy();
    }
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

int Strategy::gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

long long Strategy::divide_and_round_up(long long x, long long y) {
    return 1 + ((x - 1) / y);
}

int Strategy::next_multiple_of(int n_to_round, int multiple) {
    if (multiple == 0)
        return n_to_round;

    int remainder = n_to_round % multiple;
    if (remainder == 0)
        return n_to_round;

    return n_to_round + multiple - remainder;
}

// find all divisors of a given number n
std::vector<int> Strategy::find_divisors(int n) {
    std::vector<int> divs;
    for (int i = 1; i < n; ++i) {
        if (n % i == 0) {
            divs.push_back(i);
        }
    }
    return divs;
}

std::tuple<int, int, int> Strategy::balanced_divisors(long long m, long long n, long long k, int P) {
    // sort the dimensions 
    std::vector<long long> dimensions = {m, n, k};
    std::sort(dimensions.begin(), dimensions.end());

    // find divm, divn, divk such that m/divm = n/divn = k/divk (as close as possible)
    // be careful when dividing, since the product mnk can be very large
    double target_tile_size = std::cbrt(1.0*dimensions[1]*dimensions[2] / P * dimensions[0]);
    int divk = closest_divisor(P, this->k, target_tile_size);
    P /= divk;
    int divn = closest_divisor(P, this->n, target_tile_size);
    P /= divn;
    int divm = P;

    return std::make_tuple(divm, divn, divk);
}

// find all prime factors of a given number n
std::vector<int> Strategy::decompose(int n) {
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

int Strategy::closest_divisor(int P, int dimension, double target) {
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

long long Strategy::initial_memory(long long m, long long n, long long k, int P) {
    return divide_and_round_up(m*n,P) + divide_and_round_up(k*n,P) 
        + divide_and_round_up(m*k,P);
}

void Strategy::default_strategy() {
    std::vector<int> factors = decompose(P);
    long long m = this->m;
    long long n = this->n;
    long long k = this->k;
    int P = this->P;

    long long needed_memory = initial_memory(m, n, k, P);

    if (memory_limit < needed_memory) {
        throw_exception(std::string("This multiplication requires the memory ")
                + "for at least " + std::to_string(needed_memory)
                + " units, but only " + std::to_string(memory_limit)
                + " units are allowed. Either increase the memory limit "
                + "or change the strategy by using more Sequential (DFS) "
                + "steps.");
    }

    for (int i = 0; i < factors.size(); ++i) {
        bool did_bfs = false;
        int accumulated_div = 1;
        int next_div = factors[i];

        // m largest => split it
        while (m/accumulated_div >= std::max(n, k) 
              && needed_memory + divide_and_round_up(k*n*next_div,P) <= memory_limit) {
            accumulated_div = next_div;
            did_bfs = true;
            ++i;
            if (i >= factors.size()) break;
            next_div *= factors[i];
        }

        if (did_bfs) {
            i--;
            step_type += "b";
            split_dimension += "m";
            divisors.push_back(accumulated_div);
            needed_memory += divide_and_round_up(k*n*accumulated_div,P);
            m /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // n largest => split it
        while (n/accumulated_div >= std::max(m, k) 
              && needed_memory + divide_and_round_up(k*m*next_div,P) <= memory_limit) {
            accumulated_div = next_div;
            did_bfs = true;
            ++i;
            if (i >= factors.size()) break;
            next_div *= factors[i];
        }

        if (did_bfs) {
            i--;
            step_type += "b";
            split_dimension += "n";
            divisors.push_back(accumulated_div);
            needed_memory += divide_and_round_up(k*m*accumulated_div,P);
            n /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // k largest => split it
        while (k/accumulated_div >= std::max(m, n) 
              && needed_memory + divide_and_round_up(n*m*next_div,P) <= memory_limit) {
            accumulated_div = next_div;
            did_bfs = true;
            ++i;
            if (i >= factors.size()) break;
            next_div *= factors[i];
        }

        if (did_bfs) {
            i--;
            step_type += "b";
            split_dimension += "k";
            divisors.push_back(accumulated_div);
            needed_memory += divide_and_round_up(m*n*accumulated_div,P);
            k /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // if BFS steps were not possible
        // then perform DFS step first
        if (!did_bfs) {
            // don't count this iteration
            i--;
            step_type += "d";
            int div = 2;
            divisors.push_back(div);

            // if m largest => split it
            if (m >= std::max(k, n)) {
                split_dimension += "m";
                m /= div;
                continue;
            }

            // if n largest => split it
            if (n >= std::max(m, k)) {
                split_dimension += "n";
                n /= div;
                continue;
            }

            // if k largest => split it
            if (k >= std::max(m, n)) {
                split_dimension += "k";
                k /= div;
                continue;
            }
        }
    }
}

void Strategy::square_strategy() {
    long long m = this->m;
    long long n = this->n;
    long long k = this->k;
    int P = this->P;

    long long needed_memory = initial_memory(m, n, k, P);

    if (memory_limit < needed_memory) {
        throw_exception(std::string("This multiplication requires the memory ")
                + "for at least " + std::to_string(needed_memory)
                + " units, but only " + std::to_string(memory_limit)
                + " units are allowed. Either increase the memory limit "
                + "or change the strategy by using more Sequential (DFS) "
                + "steps.");
    }

    while (P > 1) {
        int divm, divn, divk;
        std::tie(divm, divn, divk) = balanced_divisors(m, n, k, P);

        // find prime factors of divm, divn, divk
        std::vector<int> divm_factors = decompose(divm);
        std::vector<int> divn_factors = decompose(divn);
        std::vector<int> divk_factors = decompose(divk);

        int mi, ni, ki;
        mi = ni = ki = 0;

        int total_divisors = divm_factors.size() + divn_factors.size()
            + divk_factors.size();

        // Iterate through all prime factors of divm, divn and divk and 
        // divide each dimensions with corresponding prime factors as long
        // as that dimension is the largest one.
        // Instead of dividing immediately m/divm, n/divn and k/divk,
        // it's always better to divide the dimension with smaller factors first
        // that are large enough to make that dimension NOT be the largest one after division
        for (int i = 0; i < total_divisors; ++i) {
            int accumulated_div = 1;
            int next_div = 1;

            bool did_bfs = false;

            if (mi < divm_factors.size()) {
                next_div = divm_factors[mi];
            }

            // if m largest => split it
            while (mi < divm_factors.size() && m/accumulated_div >= std::max(n, k)
                  && needed_memory + divide_and_round_up(k*n*next_div,P) <= memory_limit) {
                accumulated_div = next_div;
                did_bfs = true;
                mi++;
                i++;
                if (mi >= divm_factors.size() || i >= total_divisors) break;
                next_div *= divm_factors[mi];
            }

            if (did_bfs) {
                i--;
                needed_memory += divide_and_round_up(k*n*accumulated_div,P);
                split_dimension += "m";
                step_type += "b";
                divisors.push_back(accumulated_div);
                m /= accumulated_div;
                P /= accumulated_div;
                continue;
            }

            if (ni < divn_factors.size()) {
                next_div = divn_factors[ni];
            }

            // if n largest => split it
            while (ni < divn_factors.size() && n/accumulated_div >= std::max(m, k)
                  && needed_memory + divide_and_round_up(k*m*next_div,P) <= memory_limit) {
                accumulated_div = next_div;
                did_bfs = true;
                ni++;
                i++;
                if (ni >= divn_factors.size() || i >= total_divisors) break;
                next_div *= divn_factors[ni];
            }

            if (did_bfs) {
                i--;
                needed_memory += divide_and_round_up(k*m*accumulated_div,P);
                split_dimension += "n";
                step_type += "b";
                divisors.push_back(accumulated_div);
                n /= accumulated_div;
                P /= accumulated_div;
                continue;
            }

            if (ki < divk_factors.size()) {
                next_div = divk_factors[ki];
            }

            // if k largest => split it
            while (ki < divk_factors.size() && k/accumulated_div >= std::max(m, n)
                  && needed_memory + divide_and_round_up(n*m*next_div,P) <= memory_limit) {
                accumulated_div = next_div;
                did_bfs = true;
                ki++;
                i++;
                if (ki >= divk_factors.size() || i >= total_divisors) break;
                next_div *= divk_factors[ki];
            }

            if (did_bfs) {
                i--;
                needed_memory += divide_and_round_up(n*m*accumulated_div,P);
                split_dimension += "k";
                step_type += "b";
                divisors.push_back(accumulated_div);
                k /= accumulated_div;
                P /= accumulated_div;
                continue;
            }

            // if not enough memory for any of the proposed BFS steps
            // then perform a single DFS step and recompute again
            // best divm, divn, divk for the smaller problem
            if (!did_bfs) {
                // don't count this iteration
                i--;
                step_type += "d";
                int div = 2;
                divisors.push_back(div);

                // if m largest => split it
                if (m >= std::max(k, n)) {
                    split_dimension += "m";
                    m /= div;
                    break;
                }

                // if n largest => split it
                if (n >= std::max(m, k)) {
                    split_dimension += "n";
                    n /= div;
                    break;
                }

                // if k largest => split it
                if (k >= std::max(m, n)) {
                    split_dimension += "k";
                    k /= div;
                    break;
                }
            }
        }
    }
}

void Strategy::spartition_strategy() {
    spartition::ProblemParameters params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.divStrat = spartition::DivisionStrategy::recursive;
    params.P = P;
    params.S = memory_limit;
    params.schedule = spartition::schedType::S3D;
    spartition::Schedule schedule = spartition::GenerateSchedule(params);

    this->P = schedule.numTilesM * schedule.numTilesN * schedule.numTilesK;

    for (auto step : schedule.divisions) {
        if (step.Dim == spartition::dim::dimM) {
            split_dimension += "m";
        } else if (step.Dim == spartition::dim::dimN) {
            split_dimension += "n";
        } else {
            split_dimension += "k";
        }

        divisors.push_back(step.SplitSize);

        if (step.SplitType == spartition::splitType::BFS)
            step_type += "b";
        else
            step_type += "d";
    }
}

// token is a triplet e.g. bm3 (denoting BFS (m / 3) step)
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
int Strategy::find_flag(const std::string& short_flag, const std::string& long_flag, 
        const std::string& message, const std::string& line, bool compulsory) {
    auto position = line.find(short_flag);
    auto length = short_flag.length();
    if (position == std::string::npos) {
        position = line.find(long_flag);
        length = long_flag.length();
        if (position == std::string::npos) {
            if (compulsory)
                throw_exception(message);
            else
                return -1;
        }
    }
    while (line[position + length] == ' ') length++;
    return position + length;
}

// looks for the defined flag in the line
// if found return true, otherwise returns false
bool Strategy::flag_exists(const std::string& short_flag, const std::string& long_flag, 
        const std::string& line) {
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
int Strategy::next_int(int start, const std::string& line) {
    if (start < 0) 
        return -1;
    std::regex int_expr("([0-9]+)");
    auto it = std::sregex_iterator(line.begin() + start, line.end(), int_expr);
    int result = std::stoi(it->str());
    return result;
}

// finds the next int after start in the line
long long Strategy::next_long_long(int start, const std::string& line) {
    if (start < 0) 
        return -1;
    std::regex int_expr("([0-9]+)");
    auto it = std::sregex_iterator(line.begin() + start, line.end(), int_expr);
    long long result = std::stoll(it->str());
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

long long Strategy::required_memory(Strategy& strategy) {
    long long m = strategy.m;
    long long n = strategy.n;
    long long k = strategy.k;
    long long P = strategy.P;

    long long initial_size = initial_memory(m, n, k, P);

    for (int step = 0; step < strategy.n_steps; ++step) {
        int div = strategy.divisor(step);

        if (strategy.bfs_step(step)) {
            if (strategy.split_m(step))
                initial_size += divide_and_round_up(k*n*div,P);
            else if (strategy.split_n(step))
                initial_size += divide_and_round_up(m*k*div,P);
            else
                initial_size += divide_and_round_up(m*n*div,P);
            P /= div;
        }

        m /= strategy.divisor_m(step);
        n /= strategy.divisor_n(step);
        k /= strategy.divisor_k(step);
    }

    return initial_size;
}

// checks if the strategy is well-defined
void Strategy::check_if_valid() {
    int mi = m;
    int ni = n;
    int ki = k;
    int Pi = P;

    for (size_t i = 0; i < n_steps; ++i) {
        if (divisors[i] <= 1) {
            throw_exception(std::string("Divisors in each step must be larger than 1.")
                    + "Divisor in step " + std::to_string(i) + " = " 
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
            n_bfs_steps++;
            if (Pi <= 1) {
                throw_exception(std::string("Not enough processors for this division strategy.")
                        + "The product of all divisors in a BFS step should be equal "
                        + "to the number of processors");
            }

            if (Pi % divisors[i] != 0) {
                throw_exception(std::string("The number of processors left in each BFS step ")
                        + "should be divisible by divisor.");
            }

            Pi /= divisors[i];
        } else {
            n_dfs_steps++;
        }

        if (split_dimension[i] == 'm') {
            mi /= divisors[i];
        }

        // since we are using column major ordering, the #columns of each matrix must be at least
        // the number of processors left at that step
        if (split_dimension[i] == 'n') {
            ni /= divisors[i];
            if (ni < Pi) {
                throw_exception(std::string("Dimension n at step ") + std::to_string(i) + " = "
                        + std::to_string(ni) + ", which is less than the number of processors left = "
                        + std::to_string(Pi));
            }
        }

        if (split_dimension[i] == 'k') {
            ki /= divisors[i];
            if (ki < Pi) {
                throw_exception(std::string("Dimension k at step ") + std::to_string(i) + " = "
                        + std::to_string(ki) + ", which is less than the number "
                        + "of processors left = "
                        + std::to_string(Pi));
            }
        }
    }
    if (Pi != 1) {
        throw_exception(std::string("Too many processors. The number of processors should be ")
                + "equal to the product of divisors in all BFS steps.");
    }

    memory_used = required_memory(*this);
    // check if we have enough memory for this splitting strategy
    if (memory_limit < memory_used) {
        throw_exception(std::string("The splitting strategy requires memory for roughly ")
                + std::to_string(memory_used) + " elements, but the memory limit is only " 
                + std::to_string(memory_limit) + " elements. Either increase the memory limit "
                + "or change the strategy. (Hint: you could use some sequential (DFS) "
                + "steps to decrease the required memory.)");
    }
}

std::ostream& operator<<(std::ostream& os, const Strategy& other) {
    os << "Matrix dimensions (m, n, k) = (" << other.m << ", " << other.n << ", " 
        << other.k << ")\n";
    os << "Number of processors: " << other.P << "\n";
    if (other.topology) {
        os << "Communication-aware topology turned on.\n";
    }
    os << "Divisions strategy: \n";
    for (size_t i = 0; i < other.n_steps; ++i) {
        if (other.step_type[i] == 'b') {
            os << "BFS (" << other.split_dimension[i] << " / " << other.divisors[i] << ")\n";
        } else {
            os << "DFS (" << other.split_dimension[i] << " / " << other.divisors[i] << ")\n";
        }
    }
    std::cout << "Required memory per rank (in #elements): " << other.memory_used << "\n";
    return os;
}
