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
    // default_strategy();
    // spartition_strategy();
    square_strategy();
    n_steps = divisors.size();
    check_if_valid();
}

void Strategy::initialize(const std::string& cmd_line) {
    auto m_it = options::find_flag("-m", "--m_dimension", "Dimension m has to be defined.", cmd_line);
    auto n_it = options::find_flag("-n", "--n_dimension", "Dimension n has to be defined.", cmd_line);
    auto k_it = options::find_flag("-k", "--k_dimension", "Dimension k has to be defined.", cmd_line);
    auto P_it = options::find_flag("-P", "--processors", 
            "Number of processors has to be defined.", cmd_line);
    auto M_it = options::find_flag("-L", "--memory", 
            "Memory limit: maximum number of elements each rank can own.", cmd_line, false);
    m = options::next_int(m_it, cmd_line);
    n = options::next_int(n_it, cmd_line);
    k = options::next_int(k_it, cmd_line);
    P = options::next_int(P_it, cmd_line);
    memory_limit = options::next_long_long(M_it, cmd_line);

    // if memory limit not given, assume we have infinity
    // (i.e. assume that each rank can store all 3 matrices)
    if (memory_limit < 0) {
        memory_limit = std::numeric_limits<long long>::max();
    }
    topology = options::flag_exists("-t", "--topology", cmd_line);

    one_sided_communication = options::flag_exists("-o", "--one_sided_communication", cmd_line);

    bool steps_predefined = options::flag_exists("-s", "--steps", cmd_line);

    if (steps_predefined) {
        auto steps_it = options::find_flag("-s", "--steps", "Division steps have to be defined.", cmd_line);
        process_steps(steps_it, cmd_line);
    }
    else {
        // default_strategy();
        // spartition_strategy();
        square_strategy();
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

long long Strategy::initial_memory(long long m, long long n, long long k, int P) {
    return math_utils::divide_and_round_up(m*n,P) + math_utils::divide_and_round_up(k*n,P) 
        + math_utils::divide_and_round_up(m*k,P);
}

void Strategy::default_strategy() {
    std::vector<int> factors = math_utils::decompose(P);
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

    std::cout << "Default strategy" << std::endl;

    for (int i = 0; i < factors.size(); ++i) {
        bool did_bfs = false;
        int accumulated_div = 1;
        int next_div = factors[i];

        // m largest => split it
        while (m/accumulated_div >= std::max(n, k) 
              && needed_memory + math_utils::divide_and_round_up(k*n*next_div,P) <= memory_limit) {
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
            needed_memory += math_utils::divide_and_round_up(k*n*accumulated_div,P);
            m /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // n largest => split it
        while (n/accumulated_div >= std::max(m, k) 
              && needed_memory + math_utils::divide_and_round_up(k*m*next_div,P) <= memory_limit) {
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
            needed_memory += math_utils::divide_and_round_up(k*m*accumulated_div,P);
            n /= accumulated_div;
            P /= accumulated_div;
            continue;
        }

        // k largest => split it
        while (k/accumulated_div >= std::max(m, n) 
              && needed_memory + math_utils::divide_and_round_up(n*m*next_div,P) <= memory_limit) {
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
            needed_memory += math_utils::divide_and_round_up(m*n*accumulated_div,P);
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
    split_dimension = "";
    step_type = "";
    divisors.clear();

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
        std::tie(divm, divn, divk) = math_utils::balanced_divisors(m, n, k, P);

        // find prime factors of divm, divn, divk
        std::vector<int> divm_factors = math_utils::decompose(divm);
        std::vector<int> divn_factors = math_utils::decompose(divn);
        std::vector<int> divk_factors = math_utils::decompose(divk);

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
                  && needed_memory + math_utils::divide_and_round_up(k*n*next_div,P) <= memory_limit) {
                accumulated_div = next_div;
                did_bfs = true;
                mi++;
                i++;
                if (mi >= divm_factors.size() || i >= total_divisors) break;
                next_div *= divm_factors[mi];
            }

            if (did_bfs) {
                i--;
                needed_memory += math_utils::divide_and_round_up(k*n*accumulated_div,P);
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
                  && needed_memory + math_utils::divide_and_round_up(k*m*next_div,P) <= memory_limit) {
                accumulated_div = next_div;
                did_bfs = true;
                ni++;
                i++;
                if (ni >= divn_factors.size() || i >= total_divisors) break;
                next_div *= divn_factors[ni];
            }

            if (did_bfs) {
                i--;
                needed_memory += math_utils::divide_and_round_up(k*m*accumulated_div,P);
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
                  && needed_memory + math_utils::divide_and_round_up(n*m*next_div,P) <= memory_limit) {
                accumulated_div = next_div;
                did_bfs = true;
                ki++;
                i++;
                if (ki >= divk_factors.size() || i >= total_divisors) break;
                next_div *= divk_factors[ki];
            }

            if (did_bfs) {
                i--;
                needed_memory += math_utils::divide_and_round_up(n*m*accumulated_div,P);
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

    std::string step_type_shorter = "";
    std::string split_dimension_shorter = "";
    std::vector<int> divisors_shorter;

    for (int i = 0; i < divisors.size(); ++i) {
        if (step_type[i] == 'b') {
            step_type_shorter += "b";
            split_dimension_shorter += split_dimension[i];
            divisors_shorter.push_back(divisors[i]);
            continue;
        }

        int j = i;
        int divm = 1;
        int divn = 1;
        int divk = 1;

        while (step_type[j] == 'd') {
            if (split_dimension[j] == 'm') 
                divm *= divisors[j];
            else if (split_dimension[j] == 'n')
                divn *= divisors[j];
            else 
                divk *= divisors[j];
            j++;
        }

        if (divm > 1) {
            split_dimension_shorter += "m";
            step_type_shorter += "d";
            divisors_shorter.push_back(divm);
        }
        if (divn > 1) {
            split_dimension_shorter += "n";
            step_type_shorter += "d";
            divisors_shorter.push_back(divn);
        }
        if (divk > 1) {
            split_dimension_shorter += "k";
            step_type_shorter += "d";
            divisors_shorter.push_back(divk);
        }

        i = j - 1;
    }

    split_dimension = split_dimension_shorter;
    step_type = step_type_shorter;
    divisors = divisors_shorter;
}

double communication_cost(long long M, long long N, long long K, int gridM, int gridN, int gridK) {
    int sizeM = M / gridM;
    int sizeN = N / gridN;
    int sizeK = K / gridK;
    long long tile_cost = sizeM * sizeN + sizeM * sizeK + sizeN * sizeK;
    double total_cost = 1.0 * tile_cost * gridM * gridN * gridK;
    return total_cost;
}

void processor_grid(unsigned p, 
        int M, int N, int K,
        int& gridM, int& gridN, int& gridK,
        double maxCompLoss = 0.03, double maxCommLoss = 0.2) {

    unsigned lostProcesses = p - gridM * gridN * gridK;

    // if we loose too many processes, try to find something better
    double p_ratio = 1.0 * lostProcesses / p;
    if (p_ratio > maxCompLoss) {
        double optCommCost = communication_cost(M, N, K, gridM, gridN, gridK);
        double curCommCost = std::numeric_limits<double>::max();
        unsigned gridMcurrent, gridNcurrent, gridKcurrent;
        for (unsigned i = 0; i < p * maxCompLoss; i++) {
            if (1.0 * curCommCost / optCommCost > 1 + maxCommLoss) {
                std::tie(gridMcurrent, gridNcurrent, gridKcurrent) = math_utils::balanced_divisors(M, N, K, p - i);
                curCommCost = communication_cost(M, N, K, gridMcurrent, gridNcurrent, gridKcurrent);
            }
            else {
                gridM = gridMcurrent;
                gridN = gridNcurrent;
                gridK = gridKcurrent;
                break;
            }
        }
    }
}

void Strategy::spartition_strategy() {
    double load = std::cbrt(k) * std::cbrt(m) * std::cbrt(n) / std::cbrt(P);
    int a = (int)std::min(std::sqrt(memory_limit), load);
    int b = (int)std::max(1.0 * m * n * k / (P * memory_limit), load);

    int n_tiles_m = (m - 1) / a + 1;
    int n_tiles_n = (n - 1) / a + 1;
    int n_tiles_k = (k - 1) / b + 1;

    int tile_size_m = (m - 1) / n_tiles_m + 1;
    int tile_size_n = (n - 1) / n_tiles_n + 1;
    int tile_size_k = (k - 1) / n_tiles_k + 1;

    while (n_tiles_m * n_tiles_n * n_tiles_k > P) {
        if (a < std::sqrt(memory_limit)) {
            //find which dimension requires least stretching 
            int new_tile_size_m = n_tiles_m == 1 ? std::numeric_limits<int>::max() : (m - 1) / (n_tiles_m - 1) + 1;
            int new_tile_size_n = n_tiles_n == 1 ? std::numeric_limits<int>::max() : (n - 1) / (n_tiles_n - 1) + 1;
            int new_tile_size_k = n_tiles_k == 1 ? std::numeric_limits<int>::max() : (k - 1) / (n_tiles_k - 1) + 1;
            if (new_tile_size_k <= std::min(new_tile_size_m, new_tile_size_n)) {
                n_tiles_k = (k - 1) / new_tile_size_k + 1;
            }
            else {
                if (new_tile_size_n < new_tile_size_m && new_tile_size_n * tile_size_m < memory_limit) {
                    n_tiles_n = (n - 1) / new_tile_size_n + 1;
                }
                else if (new_tile_size_m * tile_size_n < memory_limit) {
                    n_tiles_m = (m - 1) / new_tile_size_m + 1;
                }
                else {
                    n_tiles_k = (k - 1) / new_tile_size_k + 1;
                }
            }
            if (n_tiles_m * n_tiles_n * n_tiles_k <= P) {
                break;
            }
        }
        else {
            n_tiles_k = P / (n_tiles_m * n_tiles_n);
            if (n_tiles_m * n_tiles_n * n_tiles_k <= P) {
                break;
            }
        }
    }

    //physical num cores refinement
    processor_grid(P, m, n, k, n_tiles_m, n_tiles_n, n_tiles_k);

    tile_size_m = (m - 1) / n_tiles_m + 1;
    tile_size_n = (n - 1) / n_tiles_n + 1;
    tile_size_k = (k - 1) / n_tiles_k + 1;

    double localMemCapacity = 1.0 * memory_limit / std::max(tile_size_m, tile_size_n);

    std::vector<int> divisionsM = math_utils::decompose(n_tiles_m);
    std::vector<int> divisionsN = math_utils::decompose(n_tiles_n);
    std::vector<int> divisionsK = math_utils::decompose(n_tiles_k);

    // Find the optimal sequential schedule
    // squareSide = std::floor(std::sqrt(memory_limit + 1.0) - 1);
    int cubicSide = std::floor(std::sqrt(memory_limit / 3.0));
    a = std::min(cubicSide, tile_size_m);
    b = std::min(cubicSide, tile_size_n);
    int c = std::min(cubicSide, tile_size_k);

    int local_m = a;
    int local_n = b;
    int local_k = c;

    // We only do sequential schedule in K dimension
    int div = (local_k - 1) / c + 1;

    if (div > 1) {
        divisors.push_back(div);
        step_type += "d";
        split_dimension += "k";
        tile_size_m = local_m;
        tile_size_n = local_n;
        tile_size_k = local_k;
    }

    // Add parallel schedule
    int max_num_divisors = std::max(divisionsM.size(),
            std::max(divisionsN.size(), divisionsK.size()));

    for (size_t i = 0; i < max_num_divisors; i++)
    {
        if (divisionsM.size() > i) {
            step_type += "b";
            split_dimension += "m";
            divisors.push_back(divisionsM[i]);
        }
        if (divisionsN.size() > i) {
            step_type += "b";
            split_dimension += "n";
            divisors.push_back(divisionsN[i]);
        }
        if (divisionsK.size() > i) {
            step_type += "b";
            split_dimension += "k";
            divisors.push_back(divisionsK[i]);
        }
    }

    P = n_tiles_m * n_tiles_n * n_tiles_k;
}

// token is a triplet e.g. bm3 (denoting BFS (m / 3) step)
void Strategy::process_token(const std::string& step_triplet) {
    if (step_triplet.length() < 3) return;
    step_type += step_triplet[0];
    split_dimension += step_triplet[1];
    divisors.push_back(options::next_int(2, step_triplet));
}

void Strategy::throw_exception(const std::string& message) {
    std::cout << "Splitting strategy not well defined.\n";
    // std::cout << *this << std::endl;
    throw std::runtime_error(message);
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
                initial_size += math_utils::divide_and_round_up(k*n*div,P);
            else if (strategy.split_n(step))
                initial_size += math_utils::divide_and_round_up(m*k*div,P);
            else
                initial_size += math_utils::divide_and_round_up(m*n*div,P);
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
#ifdef DEBUG
    std::cout << "Checking if the following strategy is valid: " << std::endl;
    std::cout << *this << std::endl;
#endif
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

        // if last step, check if #columns >= #processors that share this block of matrix
        // we only check dimensions n and k, because these are the dimensions defining
        // the number of columns, i.e. dimension m does not denote the #columns of any matrix
        if (i == n_steps - 1) {
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
    }
    if (Pi != 1) {
        throw_exception(std::string("Too many processors. The number of processors should be ")
                + "equal to the product of divisors in all BFS steps.");
    }

    memory_used = required_memory(*this);
    // check if we have enough memory for this splitting strategy
    /*
    if (memory_limit < memory_used) {
        throw_exception(std::string("The splitting strategy requires memory for roughly ")
                + std::to_string(memory_used) + " elements, but the memory limit is only " 
                + std::to_string(memory_limit) + " elements. Either increase the memory limit "
                + "or change the strategy. (Hint: you could use some sequential (DFS) "
                + "steps to decrease the required memory.)");
    }
    */
}

std::ostream& operator<<(std::ostream& os, const Strategy& other) {
    os << "Matrix dimensions (m, n, k) = (" << other.m << ", " << other.n << ", " 
        << other.k << ")\n";
    os << "Number of processors: " << other.P << "\n";
    if (other.topology) {
        os << "Communication-aware topology turned on.\n";
    }
    if (other.one_sided_communication) {
        os << "Communication backend: one-sided.\n";
    } else {
        os << "Communication backend: two-sided.\n";
    }
    os << "Divisions strategy: \n";
    for (size_t i = 0; i < other.n_steps; ++i) {
        if (other.step_type[i] == 'b') {
            os << "BFS (" << other.split_dimension[i] << " / " << other.divisors[i] << ")\n";
        } else {
            os << "DFS (" << other.split_dimension[i] << " / " << other.divisors[i] << ")\n";
        }
    }
    os << "Required memory per rank (in #elements): " << other.memory_used << "\n";
    os << "Available memory per rank (in #elements): " << other.memory_limit << "\n";
    return os;
}
