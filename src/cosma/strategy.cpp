#include <cosma/strategy.hpp>

namespace cosma {
int Strategy::min_dim_size = 1000;

std::size_t Strategy::n_steps() const {
    return divisors.size();
}

// constructors
Strategy::Strategy() = default;
// copy constructor
Strategy::Strategy(Strategy& other) = default;
Strategy::Strategy(const Strategy& other) = default;

// == operator
bool Strategy::operator==(const Strategy &other) const {
    return 
        this->m == other.m
        &&
        this->n == other.n
        &&
        this->k == other.k
        &&
        this->P == other.P
        &&
        this->memory_limit == other.memory_limit
        &&
        this->divisors == other.divisors
        &&
        this->step_type == other.step_type
        &&
        this->split_dimension == other.split_dimension
        &&
        this->overlap_comm_and_comp == other.overlap_comm_and_comp;
}

bool Strategy::operator!=(const Strategy &other) const {
    return !(*this == other);
}

// if the strategy proposed in divs, dims and types is not complete
// (i.e. does not divide the problem completely), then
// the strategy will try to use the proposed incomplete strategy
// as the prefix and will try to complete it here.
Strategy::Strategy(int mm,
                   int nn,
                   int kk,
                   size_t PP,
                   std::vector<int> &divs,
                   std::string &dims,
                   std::string &types,
                   long long mem_limit,
                   bool top,
                   bool overlap,
                   bool busy_waiting)
    : m(mm)
    , n(nn)
    , k(kk)
    , P(PP)
    , memory_limit(mem_limit)
    , divisors(divs)
    , split_dimension(dims)
    , step_type(types)
    , topology(top)
    , overlap_comm_and_comp(overlap)
    , use_busy_waiting(busy_waiting) {

    // if divisors are non-empty,
    // then take it as a prefix to this strategy
    bool incomplete_strategy = false;
    square_strategy(incomplete_strategy);
    check_if_valid();
    check_if_irregular();
    compute_min_sizes();
}

Strategy::Strategy(int mm,
                   int nn,
                   int kk,
                   size_t PP,
                   long long mem_limit,
                   bool top,
                   bool overlap,
                   bool busy_waiting)
    : m(mm)
    , n(nn)
    , k(kk)
    , P(PP)
    , memory_limit(mem_limit)
    , topology(top)
    , overlap_comm_and_comp(overlap)
    , use_busy_waiting(busy_waiting) {
    divisors.clear();
    step_type = "";
    split_dimension = "";
    bool incomplete_strategy;
    square_strategy(incomplete_strategy);
    // compress_steps();
    // optimize_strategy();
    check_if_valid();
    check_if_irregular();
    compute_min_sizes();
}

std::tuple<long long, long long, long long>
Strategy::initial_memory(long long m, long long n, long long k, int P) {
    return {
            math_utils::divide_and_round_up(m * k, P),
            math_utils::divide_and_round_up(k * n, P),
            math_utils::divide_and_round_up(m * n, P)
    };
}

bool Strategy::add_step(long long& prev_m, long long& prev_n, long long& prev_k, 
                        int& prev_P, char step, char dim_label, int divisor) {
    long long *dim1, *dim2, *dim3;
    if (dim_label == 'm') {
        dim1 = &prev_m;
        dim2 = &prev_n;
        dim3 = &prev_k;
    } else if (dim_label == 'n') {
        dim1 = &prev_n;
        dim2 = &prev_m;
        dim3 = &prev_k;
    } else {
        dim1 = &prev_k;
        dim2 = &prev_m;
        dim3 = &prev_n;
    }
    // if dimension becomes too small
    // try to correct it (by finding the smaller divisor)
    // or completely ignore this step 
    // if such a divisor cannot be found
    if (*dim1/divisor < min_dim_size) {
        // try to find smaller divisor
        int new_d = *dim1 / min_dim_size;
        // check if this divisor is feasible
        if (new_d > 1 && *dim1 / new_d >= min_dim_size) {
            split_dimension += dim_label;
            step_type += step;
            divisors.push_back(new_d);
            *dim1 /= new_d;
            // decrease the number of processes
            // by exchanging the divisor d with new_d
            if (step == 'p') {
                // change the global P as well, because the global
                // number of processors has to be decreased as well
                P = P / divisor * new_d;
                // let it look like we performed this step
                prev_P = prev_P / divisor * new_d;
            }
            return true;
        } else {
            // exclude this divisor
            // ignore this step
            if (step == 'p') {
                // change the global P as well, because the global
                // number of processors has to be decreased as well
                P = P / divisor;
                // let it look like we performed this step
                prev_P = prev_P / divisor;
            }
            return false;
        }
    } else {
        split_dimension += dim_label;
        step_type += step;
        divisors.push_back(divisor);
        *dim1 /= divisor;
        // decrease the number of processes
        // by exchanging the divisor d with new_d
        if (step == 'p') {
            // do not change the global number of processors in this case
            prev_P = prev_P / divisor;
        }
        return true;
    }
}


bool Strategy::divide(std::vector<int> &div_factors,
                      int &dim_i,
                      long long &m,
                      long long &n,
                      long long &k,
                      int &P,
                      const char label) {
    long long dim1, dim2, dim3;
    if (label == 'm') {
        dim1 = m;
        dim2 = n;
        dim3 = k;
    } else if (label == 'n') {
        dim1 = n;
        dim2 = m;
        dim3 = k;
    } else {
        dim1 = k;
        dim2 = m;
        dim3 = n;
    }

    int next_div = 1;
    int accumulated_div = 1;
    bool did_parallel = false;

    if (dim_i < div_factors.size()) {
        next_div = div_factors[dim_i];
    }

    bool largest = dim1 >= std::max(dim2, dim3);
    bool first_run = true;

    // std::cout << "m-split divide and round = " <<
    // math_utils::divide_and_round_up(k * n * next_div, P) << std::endl;
    // std::cout << "m / acc_div = " << m/accumulated_div << std::endl;
    // if m largest => split it
    while (dim_i < div_factors.size() && (largest || first_run)) {
        accumulated_div = next_div;
        did_parallel = true;
        dim_i++;
        // i++;
        if (dim_i >= div_factors.size())
            break;
        next_div *= div_factors[dim_i];

        first_run = false;
        largest = dim1 / accumulated_div >= std::max(dim2, dim3);
    }

    if (did_parallel) {
        // i--;
        return add_step(m, n, k, P, 'p', label, accumulated_div);
    }

    return did_parallel;
}

std::tuple<long long, long long, long long>
maximum_memory(long long m, long long n, long long k, 
               int divm, int divn, int divk, int P) {
    using dim_pair = std::tuple<long long, int, char>;
    std::vector<dim_pair> dims = {{m, divm, 'B'}, {n, divn, 'A'}, {k, divk, 'C'}};
    std::sort(dims.begin(), dims.end(),
              [](const dim_pair& a, const dim_pair& b) -> bool {
                  return std::get<0>(a) > std::get<0>(b) ||
                         std::get<0>(a) == std::get<0>(b) && 
                         std::get<1>(a) < std::get<1>(b);
              }
    );

    long long memory_A = 0;
    long long memory_B = 0;
    long long memory_C = 0;

    for (std::size_t i = 0; i < dims.size(); ++i) {
        auto& dim = dims[i];
        auto div = std::get<1>(dim);
        if (div > 1) {
            auto& next_dim = dims[(i+1) % 3];
            auto& next_next_dim = dims[(i+2) % 3];
            auto copied_matrix_size = std::get<0>(next_dim) * std::get<0>(next_next_dim);
            auto memory = math_utils::divide_and_round_up(copied_matrix_size * div, P);

            auto label = std::get<2>(dim);
            if (label == 'A') {
                memory_A = memory;
            } else if (label == 'B') {
                memory_B = memory;
            } else {
                memory_C = memory;
            }
            P /= div;
            std::get<0>(dim) /= div;
        }
    }
    return {memory_A, memory_B, memory_C};
}

long long memory_with_buffer_optimization(std::vector<long long>& memory_A,
                                          std::vector<long long>& memory_B,
                                          std::vector<long long>& memory_C) {
    long long total_memory = 0;

    // sort (descreasingly) memory for each communication round
    std::sort(memory_A.rbegin(), memory_A.rend());
    std::sort(memory_B.rbegin(), memory_B.rend());
    std::sort(memory_C.rbegin(), memory_C.rend());

    // take the 2 largest elements, because the memory
    // is reused in each communication per matrix
    // memory for matrix A
    if (memory_A.size() > 0) {
        total_memory += memory_A[0];
    }
    if (memory_A.size() > 1) {
        total_memory += memory_A[1];
    }

    // memory for matrix B
    if (memory_B.size() > 0) {
        total_memory += memory_B[0];
    }
    if (memory_B.size() > 1) {
        total_memory += memory_B[1];
    }

    // memory for matrix C
    if (memory_C.size() > 0) {
        total_memory += memory_C[0];
    }
    if (memory_C.size() > 1) {
        total_memory += memory_C[1];
    }

    return total_memory;
}

void Strategy::square_strategy(bool& incomplete_strategy) {
    long long m = this->m;
    long long n = this->n;
    long long k = this->k;
    int P = this->P;

    // total needed memory
    memory_used = 0;

    // initial needed memory
    long long init_memory_A, init_memory_B, init_memory_C;
    std::tie(init_memory_A, init_memory_B, init_memory_C) = initial_memory(m, n, k, P);

    // total memory required for each matrix without buffer optimization
    std::vector<long long> memory_A = {init_memory_A};
    std::vector<long long> memory_B = {init_memory_B};
    std::vector<long long> memory_C = {init_memory_C};

    for (int i = 0; i < divisors.size(); ++i) {
        int div = divisors[i];

        if (step_type[i] == 'p') {
            if (!split_A(i)) {
                memory_A.push_back(math_utils::divide_and_round_up(m * k * div, P));
            } else if (!split_B(i)) {
                memory_B.push_back(math_utils::divide_and_round_up(k * n * div, P));
            } else {
                memory_C.push_back(math_utils::divide_and_round_up(m * n * div, P));
            }
            P /= div;
        }

        m /= divisor_m(i);
        n /= divisor_n(i);
        k /= divisor_k(i);
    }


    // if P == 1 at this point, then it means that the complete strategy was already given
    // at the beginning, so do not try to modify it further.
    incomplete_strategy = P > 1;

    if (!incomplete_strategy) {
        memory_used = memory_with_buffer_optimization(memory_A, memory_B, memory_C);

        if (memory_limit < memory_used) {
            throw_exception(
                std::string("This multiplication requires the memory ") +
                "for at least " + std::to_string(memory_used) +
                " units, but only " + std::to_string(memory_limit) +
                " units are allowed. Either increase the memory limit " +
                "or change the strategy by using more sequential " + "steps.");
        }
        return;
    }

    int divm, divn, divk;
    std::tie(divm, divn, divk) = 
        math_utils::balanced_divisors(m, n, k, P, min_dim_size);

    long long additional_memory_A = 0;
    long long additional_memory_B = 0;
    long long additional_memory_C = 0;

    std::tie(additional_memory_A, additional_memory_B, additional_memory_C) 
        = maximum_memory(m, n, k, divm, divn, divk, P);

    std::vector<long long> new_memory_A = memory_A;
    std::vector<long long> new_memory_B = memory_B;
    std::vector<long long> new_memory_C = memory_C;

    new_memory_A.push_back(additional_memory_A);
    new_memory_B.push_back(additional_memory_B);
    new_memory_C.push_back(additional_memory_C);

    // if not enough memory for all of the proposed parallel steps
    // then perform a single sequential step and recompute again
    // best divm, divn, divk for the smaller problem
    long long used =
        memory_with_buffer_optimization(new_memory_A,
                                        new_memory_B,
                                        new_memory_C);
    while (used > memory_limit) {
        int div = 2;
        bool success = false;

        if (m >= std::max(k, n)) {
            // if m largest => split it
            success = add_step(m, n, k, P, 's', 'm', div);
        } else if (n >= std::max(m, k)) {
            // if n largest => split it
            success = add_step(m, n, k, P, 's', 'n', div);
        } else {
            // if k largest => split it
            success = add_step(m, n, k, P, 's', 'k', div);
        }

        if (!success) {
            throw_exception(std::string("Not enough memory for this strategy. ")
                  + "Either decrease the min_dim_size in the strategy "
                  + "to allow dimensions to be further split OR "
                  + "increase the memory limit in the strategy "
                  + "to allow COSMA to use more memory.");
        }

        std::tie(divm, divn, divk) = 
            math_utils::balanced_divisors(m, n, k, P, min_dim_size);

        std::tie(additional_memory_A, additional_memory_B, additional_memory_C) 
            = maximum_memory(m, n, k, divm, divn, divk, P);

        new_memory_A = memory_A;
        new_memory_B = memory_B;
        new_memory_C = memory_C;

        new_memory_A.push_back(additional_memory_A);
        new_memory_B.push_back(additional_memory_B);
        new_memory_C.push_back(additional_memory_C);
        used =
            memory_with_buffer_optimization(new_memory_A,
                                            new_memory_B,
                                            new_memory_C);
    }

    memory_used = used;

    P = divm * divn * divk;

    // find prime factors of divm, divn, divk
    std::vector<int> divm_factors = math_utils::decompose(divm);
    std::vector<int> divn_factors = math_utils::decompose(divn);
    std::vector<int> divk_factors = math_utils::decompose(divk);

    int mi, ni, ki;
    mi = ni = ki = 0;

    int total_divisors =
        divm_factors.size() + divn_factors.size() + divk_factors.size();

    // Iterate through all prime factors of divm, divn and divk and
    // divide each dimension with corresponding prime factors as long
    // as that dimension is the largest one.
    // Instead of dividing immediately m/divm, n/divn and k/divk,
    // it's always better to divide the dimension with smaller factors first
    // that are large enough to make that dimension NOT be the largest one
    // after division
    while (mi + ni + ki < total_divisors) {
        int i = mi + ni + ki;
        bool did_parallel = false;

        long long mm = mi >= divm_factors.size() ? 1 : m;
        long long nn = ni >= divn_factors.size() ? 1 : n;
        long long kk = ki >= divk_factors.size() ? 1 : k;

        if (mm >= std::max(nn, kk)) {
            did_parallel =
                divide(divm_factors, mi, m, n, k, P, 'm');
            if (did_parallel)
                continue;
        }

        if (nn >= std::max(mm, kk)) {
            did_parallel =
                divide(divn_factors, ni, m, n, k, P, 'n');
            if (did_parallel)
                continue;
        }

        if (kk >= std::max(mm, nn)) {
            did_parallel =
                divide(divk_factors, ki, m, n, k, P, 'k');
            if (did_parallel)
                continue;
        }

        if (!did_parallel) {
            throw_exception(std::string("Not enough memory for this strategy. ")
                  + "Either decrease the min_dim_size in the strategy "
                  + "to allow dimensions to be further split OR "
                  + "increase the memory limit in the strategy "
                  + "to allow COSMA to use more memory.");
        }
    }

    std::string step_type_shorter = "";
    std::string split_dimension_shorter = "";
    std::vector<int> divisors_shorter;
    this->P = 1;

    for (int i = 0; i < divisors.size(); ++i) {
        if (step_type[i] == 'p') {
            int div = divisors[i];
            while (i + 1 < divisors.size() && step_type[i + 1] == 'p' &&
                   split_dimension[i + 1] == split_dimension[i]) {
                div *= divisors[i + 1];
                i++;
            }
            step_type_shorter += "p";
            split_dimension_shorter += split_dimension[i];
            divisors_shorter.push_back(div);
            this->P *= div;
            continue;
        }

        int j = i;
        int divm = 1;
        int divn = 1;
        int divk = 1;

        while (step_type[j] == 's') {
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
            step_type_shorter += "s";
            divisors_shorter.push_back(divm);
        }
        if (divn > 1) {
            split_dimension_shorter += "n";
            step_type_shorter += "s";
            divisors_shorter.push_back(divn);
        }
        if (divk > 1) {
            split_dimension_shorter += "k";
            step_type_shorter += "s";
            divisors_shorter.push_back(divk);
        }

        i = j - 1;
    }

    split_dimension = split_dimension_shorter;
    step_type = step_type_shorter;
    divisors = divisors_shorter;
}

void Strategy::throw_exception(const std::string &message) {
    std::cout << "Splitting strategy not well defined.\n";
    std::cout << message << std::endl;
    std::cout << *this << std::endl;
    throw std::runtime_error(message);
}

bool Strategy::split_m(size_t i) const { return split_dimension[i] == 'm'; }

bool Strategy::split_n(size_t i) const { return split_dimension[i] == 'n'; }

bool Strategy::split_k(size_t i) const { return split_dimension[i] == 'k'; }

bool Strategy::split_A(size_t i) const { return split_m(i) || split_k(i); }

bool Strategy::split_B(size_t i) const { return split_k(i) || split_n(i); }

bool Strategy::split_C(size_t i) const { return split_m(i) || split_n(i); }

bool Strategy::split(char label, size_t step) const {
    if (label == 'A')
        return split_A(step);
    else if (label == 'B')
        return split_B(step);
    else
        return split_C(step);
}

bool Strategy::sequential_step(size_t i) const { return step_type[i] == 's'; }

bool Strategy::parallel_step(size_t i) const { return step_type[i] == 'p'; }

int Strategy::divisor(size_t i) const { return divisors[i]; }

int Strategy::divisor_m(size_t i) const { return split_m(i) ? divisors[i] : 1; }

int Strategy::divisor_n(size_t i) const { return split_n(i) ? divisors[i] : 1; }

int Strategy::divisor_k(size_t i) const { return split_k(i) ? divisors[i] : 1; }

int Strategy::divisor_row(char matrix, size_t i) const {
    if (matrix == 'A')
        return divisor_m(i);
    if (matrix == 'B')
        return divisor_k(i);
    if (matrix == 'C')
        return divisor_m(i);
    return 1;
}

int Strategy::divisor_col(char matrix, size_t i) const {
    if (matrix == 'A')
        return divisor_k(i);
    if (matrix == 'B')
        return divisor_n(i);
    if (matrix == 'C')
        return divisor_n(i);
    return 1;
}

bool Strategy::final_step(size_t i) const { return i == n_steps(); }

int Strategy::parallel_steps_before_gemm(char label) const {
    if (label == 'A')
        return n_parallel_steps_before_gemm_a;
    if (label == 'B')
        return n_parallel_steps_before_gemm_b;
    if (label == 'C')
        return n_parallel_steps_before_gemm_c;
    return -1;
}

// checks if the strategy is well-defined
void Strategy::check_if_valid() {
#ifdef DEBUG
    std::cout << "Checking if the following strategy is valid: " << std::endl;
    std::cout << *this << std::endl;
#endif
    if (empty() && P != 1) {
        throw_exception("Strategy empty but number of ranks P != 1");
    }

    int mi = m;
    int ni = n;
    int ki = k;
    int Pi = P;

    n_parallel_steps = 0;
    n_parallel_steps_before_gemm_a = 0;
    n_parallel_steps_before_gemm_b = 0;
    n_parallel_steps_before_gemm_c = 0;

    int P_a = 1;
    int P_b = 1;
    int P_c = 1;

    for (size_t i = 0; i < n_steps(); ++i) {
        if (divisors[i] <= 1) {
            throw_exception(
                std::string("Divisors in each step must be larger than 1.") +
                "Divisor in step " + std::to_string(i) + " = " +
                std::to_string(divisors[i]) + ".");
        }

        if (split_dimension[i] != 'm' && split_dimension[i] != 'n' &&
            split_dimension[i] != 'k') {
            throw_exception("Split dimension in each step must be m, n or k");
        }

        if (step_type[i] != 'p' && step_type[i] != 's') {
            throw_exception("Step type should be either p or s.");
        }

        if (step_type[i] == 'p') {
            n_parallel_steps++;
            if (!split_A(i)) {
                n_parallel_steps_before_gemm_a++;
            }
            if (!split_B(i)) {
                n_parallel_steps_before_gemm_b++;
            }
            if (!split_C(i)) {
                n_parallel_steps_before_gemm_c++;
            }

            if (Pi <= 1) {
                throw_exception(
                    std::string(
                        "Not enough processors for this division strategy.") +
                    "The product of all divisors in a parallel step should be "
                    "equal " +
                    "to the number of processors");
            }

            if (Pi % divisors[i] != 0) {
                throw_exception(std::string("The number of processors left in "
                                            "each parallel step ") +
                                "should be divisible by divisor.");
            }

            Pi /= divisors[i];
        } else {
            n_sequential_steps++;
            if (split_A(i)) {
                n_parallel_steps_before_gemm_a = 0;
            }
            if (split_B(i)) {
                n_parallel_steps_before_gemm_b = 0;
            }
            if (split_C(i)) {
                n_parallel_steps_before_gemm_c = 0;
            }
        }

        if (step_type[i] == 'p') {
            if (!split_A(i)) {
                P_a *= divisors[i];
            } else if (!split_B(i)) {
                P_b *= divisors[i];
            } else if (!split_C(i)) {
                P_c *= divisors[i];
            } else {
                throw_exception("Invalid strategy: In each step, some matrix has to be split.");
            }
        }

        if (split_dimension[i] == 'm') {
            mi /= divisors[i];
        } else if (split_dimension[i] == 'n') {
            ni /= divisors[i];
        } else if (split_dimension[i] == 'k') {
            ki /= divisors[i];
        } else {
            throw_exception("Unknown splitting dimension, should be m, n or k");
        }

        // if last step, check if #columns >= #processors that share this block
        // of matrix we only check dimensions n and k, because these are the
        // dimensions defining the number of columns, i.e. dimension m does not
        // denote the #columns of any matrix
        if (i == n_steps() - 1) {
            // since we are using column major ordering, the #columns of each
            // matrix must be at least the number of processors left at that
            // step
            if (ki < P_a) {
                throw_exception(std::string("Dimension k at step ") +
                                std::to_string(i) + " = " +
                                std::to_string(ki) +
                                ", which is less than the number of "
                                "processors left = " +
                                std::to_string(P_a));
            }
            if (ni < std::max(P_b, P_c)) {
                throw_exception(std::string("Dimension n at step ") +
                                std::to_string(i) + " = " +
                                std::to_string(ni) +
                                ", which is less than the number of "
                                "processors left = " +
                                std::to_string(std::min(P_b, P_c)));
            }
        }
    }
    if (Pi != 1) {
        throw_exception(
            std::string(
                "Too many processors. The number of processors should be ") +
            "equal to the product of divisors in all parallel steps.");
    }

    /*
    memory_used = required_memory(*this);
    // check if we have enough memory for this splitting strategy
    if (memory_limit < memory_used) {
        throw_exception("The splitting strategy requires memory \
                         for roughly " + std::to_string(memory_used) + " elements, \
                         but the memory limit is only " + std::to_string(memory_limit) + " elements. \
                         Either increase the memory limit or change the strategy. \
                         (Hint: you could use some sequential steps to spare some memory!)");
    }
    */
}

void Strategy::compress_steps() {
    int p_divm = 1;
    int p_divn = 1;
    int p_divk = 1;
    int s_divm = 1;
    int s_divn = 1;
    int s_divk = 1;

    for (size_t i = 0; i < split_dimension.size(); ++i) {
        if (parallel_step(i)) {
            p_divm *= divisor_m(i);
            p_divn *= divisor_n(i);
            p_divk *= divisor_k(i);
        } else {
            s_divm *= divisor_m(i);
            s_divn *= divisor_n(i);
            s_divk *= divisor_k(i);
        }
    }

    std::vector<int> divs = {p_divm, p_divn, p_divk, s_divm, s_divn, s_divk};

    divisors = std::vector<int>();
    split_dimension = "";
    step_type = "";

    for (size_t i = 0; i < divs.size(); ++i) {
        if (divs[i] > 1) {
            divisors.push_back(divs[i]);

            if (i < 3) {
                step_type += "p";
            } else {
                step_type += "s";
            }

            if (i % 3 == 0) {
                split_dimension += "m";
            } else if (i % 3 == 1) {
                split_dimension += "n";
            } else {
                split_dimension += "k";
            }
        }
    }
}

void Strategy::compute_min_sizes() {
    min_m = m;
    min_n = n;
    min_k = k;
    for (int step = 0; step < n_steps(); ++step) {
        min_m /= divisor_m(step);
        min_n /= divisor_n(step);
        min_k /= divisor_k(step);
    }
}

bool Strategy::should_overlap_comm_and_comp(int step) const {
    bool last_step = step == n_steps() - 1;
    if (!last_step) {
        return false;
    }

    int div = divisor(step);
    int divm = divisor_m(step);
    int divn = divisor_n(step);
    int divk = divisor_k(step);

    int newm = min_m;
    int newn = min_n;
    int newk = min_k;

    // overlap requires that the number of columns of the expanded matrix
    // i.e. the matrix that is not split is >= div, so that it can be split as
    // well
    bool overlap_possible = (split_m(step) && min_n >= div) ||
                            (split_n(step) && min_k >= div) ||
                            (split_k(step) && min_n >= div);

    if (split_m(step)) {
        newn /= div;
    } else if (split_n(step)) {
        newk /= div;
    } else {
        newn /= div;
    }

    bool overlap_turned_on = overlap_comm_and_comp;
    double score_no_overlap = math_utils::square_score(min_m, min_n, min_k);
    double score_with_overlap = math_utils::square_score(newm, newn, newk);
    auto diff = score_with_overlap - score_no_overlap;
    bool should_overlap = diff / score_no_overlap >= 0.5;

    // std::cout << "overlap_possible = " << overlap_possible << std::endl;
    // std::cout << "last_step = " << last_step << std::endl;
    // std::cout << "overlap_turned_on = " << overlap_turned_on << std::endl;
    // std::cout << "score_no_overlap = " << score_no_overlap << std::endl;
    // std::cout << "score_with_overlap = " << score_with_overlap << std::endl;
    // std::cout << "should_overlap = " << should_overlap << std::endl;
    bool condition = overlap_possible && overlap_turned_on && should_overlap;

#ifdef DEBUG
    std::cout << "Overlapping communication and computation." << std::endl;
#endif

    // return condition;
    return condition;
}

bool Strategy::empty() const {
    return n_steps() == 0;
}

int Strategy::n_rows(char label) const {
    if (label == 'A') 
        return m;
    if (label == 'B') 
        return k;
    if (label == 'C') 
        return m;
    return -1;
}

int Strategy::n_cols(char label) const {
    if (label == 'A') 
        return k;
    if (label == 'B') 
        return n;
    if (label == 'C') 
        return n;

    return -1;
}

// enables overlapping and updates the value of the `irregular` variable
void Strategy::enable_overlapping_comm_and_comp() {
    int last_step = n_steps() - 1;

    // if comm and comp are overlapped, then in the last step
    // the #columns of the matrix which was not split in that step
    // are being split by the same divisor to allow the overlap
    if (split_m(last_step) && min_n >= divisor_m(last_step)) {
        // overlap only possible if min_n >= divisor_m(last_step)
        overlap_comm_and_comp = true;
        if (overlap_comm_and_comp) {
            // if m is split, then B is not split and thus min_n is also split
            irregular = irregular || (min_n % divisor_m(last_step) != 0);
        }
    } else if (split_n(last_step) && min_k >= divisor_n(last_step)) {
        overlap_comm_and_comp = true;
        if (overlap_comm_and_comp) {
            // if n is split, then A is not split and thus min_k is also split
            irregular = irregular || (min_k % divisor_n(last_step) != 0);
        }
    } else if (split_k(last_step) && min_n >= divisor_k(last_step)) {
        overlap_comm_and_comp = true;
        if (overlap_comm_and_comp) {
            // if k is split, then C is not split and thus min_n is also split
            irregular = irregular || (min_n % divisor_k(last_step) != 0);
        }
    }
}

// the strategy is considered irregular if any dimension
// (at any step) is divided by a divisor that does not perfectly
// divide that dimension
void Strategy::check_if_irregular() {
    int mm = m;
    int nn = n;
    int kk = k;
    for (int i = 0; i < n_steps(); ++i) {
        if (mm % divisor_m(i) != 0) {
            irregular = true;
            return;
        }
        if (nn % divisor_n(i) != 0) {
            irregular = true;
            return;
        }
        if (kk % divisor_k(i) != 0) {
            irregular = true;
            return;
        }
        mm /= divisor_m(i);
        nn /= divisor_n(i);
        kk /= divisor_k(i);
    }
    irregular = false;
}

std::ostream &operator<<(std::ostream &os, const Strategy &other) {
    os << "Matrix dimensions (m, n, k) = (" << other.m << ", " << other.n
       << ", " << other.k << ")\n";
    os << "Number of processors: " << other.P << "\n";
    if (other.topology) {
        os << "Communication-aware topology turned on.\n";
    }
    if (other.overlap_comm_and_comp) {
        os << "Overlap of communication and computation: ON.\n";
        if (other.use_busy_waiting) {
            os << "Communication-thread policy (for overlap): "
               << "busy-waiting (using blocking one-sided MPI).\n";
        } else {
            os << "Communication-thread policy (for overlap): "
               << "polling (using non-blocking one-sided MPI).\n";
        }
    } else {
        os << "Overlap of communication and computation: OFF.\n";
    }
    os << "Divisions strategy: \n";
    for (size_t i = 0; i < other.n_steps(); ++i) {
        if (other.step_type[i] == 'p') {
            os << "parallel (" << other.split_dimension[i] << " / "
               << other.divisors[i] << ")\n";
        } else {
            os << "sequential (" << other.split_dimension[i] << " / "
               << other.divisors[i] << ")\n";
        }
    }
    os << "Required memory per rank (in #elements): " << other.memory_used
       << "\n";
    os << "Available memory per rank (in #elements): ";
    if (other.memory_limit < std::numeric_limits<long long>::max())
        os << other.memory_limit;
    else
        os << "not specified (assumed: infinite)";
    os << "\n";
    return os;
}
} // namespace cosma
