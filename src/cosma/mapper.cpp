#include <cosma/mapper.hpp>
#include <cosma/profiler.hpp>

namespace cosma {
Mapper::Mapper(char label, const Strategy &strategy, int rank)
    : label_(label)
    , strategy_(&strategy)
    , m_(strategy.n_rows(label))
    , n_(strategy.n_cols(label))
    , P_(strategy.P)
    , rank_(rank) {
    PE(preprocessing_matrices_mapper_sizes);
    skip_ranges_ = std::vector<int>(P_);
    rank_to_range_ =
        std::vector<std::vector<Interval2D>>(P_, std::vector<Interval2D>());
    mi_ = Interval(0, m_ - 1);
    ni_ = Interval(0, n_ - 1);
    Pi_ = Interval(0, P_ - 1);
    compute_sizes(mi_, ni_, Pi_, 0, strategy);
    initial_buffer_size_ = std::vector<size_t>(P_);
    range_offset_ =
        std::vector<std::vector<std::size_t>>(P_, std::vector<std::size_t>());

    for (size_t rank = 0; rank < P_; ++rank) {
        size_t size = 0;
        int matrix_id = 0;
        for (auto &matrix : rank_to_range_[rank]) {
            range_offset_[rank].push_back(size);
            size += matrix.size();
            matrix_id++;
        }
        range_offset_[rank].push_back(size);
        initial_buffer_size_[rank] = size;
        if (rank_to_range_[rank].size() == 0) {
            std::cout << "RANK " << rank << " DOES NOT OWN ANYTHING"
                      << std::endl;
        }
    }
    PL();

    PE(preprocessing_matrices_mapper_coordinates);
    // both partitions start with 0
    row_partition_set_ = std::set<int>{-1};
    col_partition_set_ = std::set<int>{-1};
    compute_range_to_rank();
    row_partition_ =
        std::vector<int>(row_partition_set_.begin(), row_partition_set_.end());
    col_partition_ =
        std::vector<int>(col_partition_set_.begin(), col_partition_set_.end());

    // compute_global_coord();
#ifdef DEBUG
    output_layout();
#endif
    PL();
    // if (rank_ >= P_) {
    //     return;
    // }
}

Mapper::Mapper(const Mapper &other)
    : label_(other.label_)
    , m_(other.m_)
    , n_(other.n_)
    , P_(other.P_)
    , rank_(other.rank_)
    , strategy_(other.strategy_)
    , rank_to_range_(other.rank_to_range_)
    , range_to_rank_(other.range_to_rank_)
    , initial_buffer_size_(other.initial_buffer_size_)
    , range_offset_(other.range_offset_)
    , mi_(other.mi_)
    , ni_(other.ni_)
    , Pi_(other.Pi_)
    , skip_ranges_(other.skip_ranges_)
    , row_partition_set_(other.row_partition_set_)
    , col_partition_set_(other.col_partition_set_)
    , row_partition_(other.row_partition_)
    , col_partition_(other.col_partition_) {
    std::lock_guard<std::mutex> guard(other.global_coord_mutex_);
    global_coord_ready_ = other.global_coord_ready_;
    global_coord_ = other.global_coord_;
}

Mapper &Mapper::operator=(const Mapper &other) {
    if (this == &other) {
        return *this;
    }

    label_ = other.label_;
    m_ = other.m_;
    n_ = other.n_;
    P_ = other.P_;
    rank_ = other.rank_;
    strategy_ = other.strategy_;
    rank_to_range_ = other.rank_to_range_;
    range_to_rank_ = other.range_to_rank_;
    initial_buffer_size_ = other.initial_buffer_size_;
    range_offset_ = other.range_offset_;
    mi_ = other.mi_;
    ni_ = other.ni_;
    Pi_ = other.Pi_;
    skip_ranges_ = other.skip_ranges_;
    row_partition_set_ = other.row_partition_set_;
    col_partition_set_ = other.col_partition_set_;
    row_partition_ = other.row_partition_;
    col_partition_ = other.col_partition_;

    std::lock(global_coord_mutex_, other.global_coord_mutex_);
    std::lock_guard<std::mutex> lock_this(global_coord_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.global_coord_mutex_,
                                           std::adopt_lock);
    global_coord_ready_ = other.global_coord_ready_;
    global_coord_ = other.global_coord_;

    return *this;
}

Mapper::Mapper(Mapper &&other) noexcept
    : label_(other.label_)
    , m_(other.m_)
    , n_(other.n_)
    , P_(other.P_)
    , rank_(other.rank_)
    , strategy_(other.strategy_)
    , rank_to_range_(std::move(other.rank_to_range_))
    , range_to_rank_(std::move(other.range_to_rank_))
    , initial_buffer_size_(std::move(other.initial_buffer_size_))
    , range_offset_(std::move(other.range_offset_))
    , mi_(other.mi_)
    , ni_(other.ni_)
    , Pi_(other.Pi_)
    , skip_ranges_(std::move(other.skip_ranges_))
    , row_partition_set_(std::move(other.row_partition_set_))
    , col_partition_set_(std::move(other.col_partition_set_))
    , row_partition_(std::move(other.row_partition_))
    , col_partition_(std::move(other.col_partition_)) {
    std::lock_guard<std::mutex> guard(other.global_coord_mutex_);
    global_coord_ready_ = other.global_coord_ready_;
    global_coord_ = std::move(other.global_coord_);
    other.global_coord_ready_ = false;
    other.global_coord_.clear();
}

Mapper &Mapper::operator=(Mapper &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    label_ = other.label_;
    m_ = other.m_;
    n_ = other.n_;
    P_ = other.P_;
    rank_ = other.rank_;
    strategy_ = other.strategy_;
    rank_to_range_ = std::move(other.rank_to_range_);
    range_to_rank_ = std::move(other.range_to_rank_);
    initial_buffer_size_ = std::move(other.initial_buffer_size_);
    range_offset_ = std::move(other.range_offset_);
    mi_ = other.mi_;
    ni_ = other.ni_;
    Pi_ = other.Pi_;
    skip_ranges_ = std::move(other.skip_ranges_);
    row_partition_set_ = std::move(other.row_partition_set_);
    col_partition_set_ = std::move(other.col_partition_set_);
    row_partition_ = std::move(other.row_partition_);
    col_partition_ = std::move(other.col_partition_);

    std::lock(global_coord_mutex_, other.global_coord_mutex_);
    std::lock_guard<std::mutex> lock_this(global_coord_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.global_coord_mutex_,
                                           std::adopt_lock);
    global_coord_ready_ = other.global_coord_ready_;
    global_coord_ = std::move(other.global_coord_);
    other.global_coord_ready_ = false;
    other.global_coord_.clear();

    return *this;
}

void Mapper::output_layout() {
    std::cout << "MATRIX " << label_ << " LAYOUT: " << std::endl;
    for (int i = 0; i < m_; ++i) {
        for (int j = 0; j < n_; ++j) {
            std::cout << local_coordinates(i, j).second << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "Row partitions:\n";
    for (auto i = 0u; i < row_partition_.size(); i++) {
        std::cout << row_partition_[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << "Column partitions:\n";
    for (auto i = 0u; i < col_partition_.size(); i++) {
        std::cout << col_partition_[i] << " ";
    }
    std::cout << std::endl << std::endl;
    ;
    /*
    std::cout << "Range to rank:\n";
    for (auto& pair : range_to_rank_) {
        std::cout << "Range " << pair.first << " is owned by rank " <<
    pair.second.first << " starting at local index " << pair.second.second <<
    std::endl;
    }
    std::cout << "\n\n";
    */

    std::cout << "Rank to range:\n";
    for (auto i = 0u; i < P_; ++i) {
        std::cout << "Rank " << i << " owns:" << std::endl;
        for (auto &range : rank_to_range_[i]) {
            std::cout << range << std::endl;
        }
        std::cout << "\n\n";
    }
    std::cout << "\n\n";
}

// finds the initial data layout
void Mapper::compute_sizes(Interval m,
                           Interval n,
                           Interval P,
                           int step,
                           const Strategy &strategy) {
    Interval2D submatrix(m, n);

    // base case
    if (strategy.final_step(step) || strategy.empty()) {
        auto submatrices = rank_to_range_[P.first()];
        rank_to_range_[P.first()].push_back(submatrix);
        return;
    }

    int divm = strategy.divisor_row(label_, step);
    int divn = strategy.divisor_col(label_, step);
    int div = strategy.divisor(step);

    // remember the previous number of fixed subranges
    // for each rank. this is only used in sequential step
    // we want the next sequential step to NOT modify the
    // subranges from the previous sequential step
    std::vector<int> prev_skip_ranges;
    if (strategy.sequential_step(step)) {
        for (int i = P.first(); i <= P.last(); ++i) {
            prev_skip_ranges.push_back(skip_ranges_[i]);
        }
    }

    for (int i = 0; i < div; ++i) {
        Interval newP = P.subinterval(div, i);
        // intervals of M, N and K that the current processor subinterval is
        // taking care of
        Interval newm = m.subinterval(divm, divm > 1 ? i : 0);
        Interval newn = n.subinterval(divn, divn > 1 ? i : 0);

        if (strategy.sequential_step(step)) {
            // invoke the substeps
            compute_sizes(newm, newn, P, step + 1, strategy);
            // skip these elements in rank_to_range_ to make the next sequential
            // step independent we assume that this many subranges are fixed in
            // this sequential step and we don't want that next sequential step
            // pop up some of the subranges stored in this sequential step (for
            // example if sequential step is followed by copy case)
            for (int rank = P.first(); rank <= P.last(); ++rank) {
                skip_ranges_[rank] = rank_to_range_[rank].size();
            }
            // don't go in other branches if dividing over absent dimension
            // it is still necessary to run the substep at least once
            // because rank_to_range_ fills up only at the end of the substep
            // and is being modified on the way back
            if (divm * divn == 1) {
                break;
            }
        } else {
            // no-copy case
            // here each substep will fill up different part of the sizes vector
            if (divm * divn > 1) {
                compute_sizes(newm, newn, newP, step + 1, strategy);
            }
            // copy case
            else {
                compute_sizes(m, n, newP, step + 1, strategy);

                for (int shift = 0; shift < newP.length(); ++shift) {
                    int rank = newP.first() + shift;

                    // go through all the submatrices this rank owns
                    auto &submatrices = rank_to_range_[rank];
                    for (int mat = skip_ranges_[rank]; mat < submatrices.size();
                         mat++) {
                        auto &matrix = submatrices[mat];

                        // and split it equally among all the ranks that
                        // this rank is communicating to in this round
                        for (int partition = 1; partition < div; ++partition) {
                            int target = partition * newP.length() + rank;
                            auto &vec = rank_to_range_[target];
                            vec.push_back(matrix.submatrix(div, partition));
                        }
                        matrix = matrix.submatrix(div, 0);
                    }
                }
                // invoke just one branch of substeps since others are the same
                // (in the copy case)
                break;
            }
        }
    }

    // if copy case is followed by a sequential step then it is necessary to not
    // permanently skip the subranges after all sequential steps since maybe the
    // first copy case wants to modify all the elements from the beginning of
    // the sequential step (to subdivide all the matrices as above)
    if (strategy.sequential_step(step)) {
        // clean after yourself, once all sequential steps have finished
        for (int i = P.first(); i <= P.last(); ++i) {
            skip_ranges_[i] = prev_skip_ranges[i - P.first()];
        }
    }
}

size_t Mapper::initial_size(int rank) const {
    // check if reorered
    // if (ranks_reordered) {
    //     rank = ranks_reordering[rank];
    // }
    if (rank < P_)
        return initial_buffer_size_[rank];
    return 0;
}

size_t Mapper::initial_size() const { return initial_size(rank_); }

std::vector<size_t> Mapper::all_initial_sizes() const {
    return initial_buffer_size_;
}

const std::vector<Interval2D> &Mapper::initial_layout(int rank) const {
    // check if reorered
    // if (ranks_reordered) {
    //     rank = ranks_reordering[rank];
    // }
    return rank_to_range_[rank];
}

const std::vector<Interval2D> &Mapper::initial_layout() const {
    return initial_layout(rank_);
}

std::vector<std::vector<Interval2D>> &Mapper::complete_layout() {
    return rank_to_range_;
}

// computes the inverse of rank_to_range_ by iterating through it
void Mapper::compute_range_to_rank() {
    for (auto rank = 0u; rank < P_; ++rank) {
        int matrix_id = 0;
        for (auto matrix : rank_to_range_[rank]) {
            range_to_rank_.insert(
                {matrix, {rank, range_offset_[rank][matrix_id]}});
            row_partition_set_.insert(matrix.rows.last());
            col_partition_set_.insert(matrix.cols.last());
            ++matrix_id;
        }
    }
}

// (gi, gj) -> (local_id, rank)
std::pair<int, int> Mapper::local_coordinates(int gi, int gj) {
    Interval row_interval;
    Interval col_interval;

    // TODO: use segment tree to locate the interval which contains (gi, gj)
    for (auto row_int = 1u; row_int < row_partition_.size(); ++row_int) {
        if (row_partition_[row_int] >= gi && row_partition_[row_int - 1] < gi) {
            row_interval = Interval(row_partition_[row_int - 1] + 1,
                                    row_partition_[row_int]);
            break;
        }
    }

    for (auto col_int = 1u; col_int < col_partition_.size(); ++col_int) {
        if (col_partition_[col_int] >= gj && col_partition_[col_int - 1] < gj) {
            col_interval = Interval(col_partition_[col_int - 1] + 1,
                                    col_partition_[col_int]);
            break;
        }
    }
    // range containing gi, gj
    Interval2D range(row_interval, col_interval);

    if (!range.contains(gi, gj)) {
        std::cout << "Error in local_coordinates(" << gi << ", " << gj
                  << ") does not belong to the range " << range << std::endl;
    }

    int rank;
    int offset;
    int local_index;

    std::tie(rank, offset) = range_to_rank_[range];
    // if (ranks_reordered) {
    //     rank = ranks_reordering[rank];
    // }
    local_index = offset + range.local_index(gi, gj);

    return {local_index, rank};
}

void Mapper::compute_global_coord() {
    const auto size = initial_size();
    global_coord_ready_ = false;
    global_coord_.assign(size, {-1, -1});

    int index = 0;
    for (auto matrix_id = 0u; matrix_id < rank_to_range_[rank_].size();
         ++matrix_id) {
        Interval2D range = rank_to_range_[rank_][matrix_id];
        for (auto local = 0; local < range.size() && index < size;
             ++local, ++index) {
            global_coord_[index] = range.global_index(local);
        }
    }
    global_coord_ready_ = true;
}

// local_id -> (gi, gj) (only for the current rank)
std::pair<int, int> Mapper::global_coordinates(int local_index) {
    if (local_index >= initial_size()) {
        return {-1, -1};
    }
    std::lock_guard<std::mutex> guard(global_coord_mutex_);
    if (!global_coord_ready_) {
        compute_global_coord();
    }
    if (local_index >= static_cast<int>(global_coord_.size())) {
        return {-1, -1};
    }
    return global_coord_[local_index];
}

// (local_id, rank) -> (gi, gj)
std::pair<int, int> Mapper::global_coordinates(int local_index, int rank) {
    // if (ranks_reordered) {
    //     rank = ranks_reordering[rank];
    // }
    // TODO: use segment tree to locate with matrix of all the matrices
    // owned by rank contain the local_index
    for (auto matrix_id = 0u; matrix_id < rank_to_range_[rank].size();
         ++matrix_id) {
        // range_offset_ returns the beginning index of matrix_id range
        // if the beginning of the matrix >= local_index then this range
        // contains local_index
        if (range_offset_[rank][matrix_id + 1] > local_index) {
            Interval2D range = rank_to_range_[rank][matrix_id];
            local_index -= range_offset_[rank][matrix_id];

            int x, y;
            std::tie(x, y) = range.global_index(local_index);
            // std::cout << "Rank " << rank << ", local_index = " << local_index
            // << " -> (" <<  x << ", " << y << ")" << std::endl;
            return {x, y};
        }
    }
    return {-1, -1};
}

char Mapper::which_matrix() { return label_; }

std::vector<std::size_t> &Mapper::local_blocks_offsets() {
    return range_offset_[rank_];
}

std::vector<Interval2D> Mapper::local_blocks() {
    if (rank_ < strategy_->P)
        return rank_to_range_[rank_];
    return {};
}

int Mapper::owner(Interval2D &block) {
    auto rank_and_offset_iterator = range_to_rank_.find(block);
    if (rank_and_offset_iterator == range_to_rank_.end()) {
        throw std::runtime_error("ERROR in mapper.cpp: the owner cannot be "
                                 "determined, the block not found.");
    }
    assert(rank_and_offset_iterator != range_to_rank_.end());
    auto rank_and_offset = rank_and_offset_iterator->second;
    auto rank = rank_and_offset.first;
    return rank;
}

costa::assigned_grid2D Mapper::get_layout_grid() {
    // **************************
    // create grid2D
    // **************************
    // prepare row intervals
    // and col intervals
    std::vector<int> rows_split;
    rows_split.reserve(row_partition_.size());
    for (const auto &tick : row_partition_) {
        rows_split.push_back(tick + 1);
    }
    std::vector<int> cols_split;
    cols_split.reserve(col_partition_.size());
    for (const auto &tick : col_partition_) {
        cols_split.push_back(tick + 1);
    }

    costa::grid2D grid(std::move(rows_split), std::move(cols_split));

    int n_blocks_row = grid.n_rows;
    int n_blocks_col = grid.n_cols;

    // **************************
    // create an assigned grid2D
    // **************************
    // create a matrix of ranks owning each block
    std::vector<std::vector<int>> owners(n_blocks_row,
                                         std::vector<int>(n_blocks_col));
    for (int i = 0; i < n_blocks_row; ++i) {
        auto r_inter = grid.row_interval(i);
        Interval row_interval(r_inter.start, r_inter.end - 1);
        for (int j = 0; j < n_blocks_col; ++j) {
            auto c_inter = grid.col_interval(j);
            Interval col_interval(c_inter.start, c_inter.end - 1);

            Interval2D range(row_interval, col_interval);
            owners[i][j] = owner(range);
        }
    }

    // create an assigned grid2D
    costa::assigned_grid2D assigned_grid(
        std::move(grid), std::move(owners), P_);

    return assigned_grid;
}

int Mapper::m() const { return m_; }

int Mapper::n() const { return n_; }

int Mapper::P() const { return P_; }

int Mapper::rank() const { return rank_; }

char Mapper::label() const { return label_; }

const Strategy &Mapper::strategy() const { return *strategy_; }

void Mapper::reorder_rank(int new_rank) {
    rank_ = new_rank;
    std::lock_guard<std::mutex> guard(global_coord_mutex_);
    global_coord_ready_ = false;
    global_coord_.clear();
}
} // namespace cosma
