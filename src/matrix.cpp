#include "matrix.hpp"

CarmaMatrix::CarmaMatrix(char label, const Strategy& strategy, int rank) :
        label_(label), rank_(rank) {
    PE(preprocessing_matrices);
    if (label_ == 'A') {
        m_ = strategy.m;
        n_ = strategy.k;
    }
    else if (label_ == 'B') {
        m_ = strategy.k;
        n_ = strategy.n;
    }
    else {
        m_ = strategy.m;
        n_ = strategy.n;
    }
    P_ = strategy.P;
    mapper_ = std::make_unique<Mapper>(label, m_, n_, P_, strategy, rank);
    layout_ = std::make_unique<Layout>(label, m_, n_, P_, rank, mapper_->complete_layout());
    matrix_ = std::vector<double>(mapper_->initial_size(rank));

    max_send_buffer_size_ = (long long) initial_size();
    max_recv_buffer_size_ = (long long) initial_size();
    compute_max_buffer_size(strategy);

    send_buffer_ = std::vector<double>(max_send_buffer_size_);
    receive_buffer_ = std::vector<double>(max_recv_buffer_size_);

    if (max_send_buffer_size_ < max_recv_buffer_size_) {
        std::cout << "less by a factor of " << 1.0 * max_recv_buffer_size_ / max_send_buffer_size_ << std::endl;
    }

    current_mat = send_buffer_.data();

    PL();
}

int CarmaMatrix::m() {
    return m_;
}

int CarmaMatrix::n() {
    return n_;
}

const int CarmaMatrix::initial_size(int rank) const {
    return mapper_->initial_size(rank);
}

const int CarmaMatrix::initial_size() const {
    return mapper_->initial_size();
}

void CarmaMatrix::compute_max_buffer_size(const Strategy& strategy) {
    Interval m(0, strategy.m - 1);
    Interval n(0, strategy.n - 1);
    Interval k(0, strategy.k - 1);
    Interval P(0, strategy.P - 1);

    compute_max_buffer_size(m, n, k, P, 0, strategy, rank_);
}

void CarmaMatrix::compute_max_buffer_size(Interval& m, Interval& n, Interval& k, Interval& P, 
    int step, const Strategy& strategy, int rank) {
    // current submatrices that are being computed
    Interval2D a_range(m, k);
    Interval2D b_range(k, n);
    Interval2D c_range(m, n);

    // For each of P processors remember which DFS bucket we are currently on
    std::vector<int> buckets = dfs_buckets(P);
    // Skip all buckets that are "before" the current submatrices. 
    // the relation submatrix1 <before> submatrix2 is defined in Interval2D.
    // Intuitively, this will skip all the buckets that are "above" or "on the left" 
    // of the current submatrices. We say "before" because whenever we split in DFS
    // sequentially, we always first start with the "above" submatrix 
    // (if the splitting is horizontal) or with the left one (if the splitting is vertical).
    // which explains the name of the relation "before".
    if (label_ == 'A') {
        update_buckets(P, a_range);
    }
    else if (label_ == 'B') {
        update_buckets(P, b_range);
    }
    else {
        update_buckets(P, c_range);
    }

    //int offset = shift(buckets[rank - P.first()]);

    // recursively invoke BFS or DFS:
    if (strategy.final_step(step)) {
        long long max_size = 0;
        if (label_ == 'A') {
            max_size = 1LL * m.length() * k.length();
        }
        else if (label_ == 'B') {
            max_size = 1LL * k.length() * n.length();
        }
        else {
            max_size = 1LL * m.length() * n.length();
        }

        if (max_size > max_recv_buffer_size_) {
            max_send_buffer_size_ = max_recv_buffer_size_;
            max_recv_buffer_size_ = max_size;
        } else if (max_size > max_send_buffer_size_) {
            max_send_buffer_size_ = max_size;
        }
    } else if (strategy.dfs_step(step)) {
        int div = strategy.divisor(step);
        int divm = strategy.divisor_m(step);
        int divn = strategy.divisor_n(step);
        int divk = strategy.divisor_k(step);

        for (int i = 0; i < div; ++i) {
            Interval newm = m.subinterval(divm, divm>1 ? i : 0);
            Interval newn = n.subinterval(divn, divn>1 ? i : 0);
            Interval newk = k.subinterval(divk, divk>1 ? i : 0);
            compute_max_buffer_size(newm, newn, newk, P, step+1, strategy, rank);

            // if dividing over absent dimension, then all the branches are the same
            // so skip the rest
            if ((label_ == 'A' && !strategy.split_A(step))
                    || (label_ == 'B' && !strategy.split_B(step))
                    || (label_ == 'C' && !strategy.split_C(step))) {
                break;
            }
        }
    } else {
        int div = strategy.divisor(step);
        int divm = strategy.divisor_m(step);
        int divn = strategy.divisor_n(step);
        int divk = strategy.divisor_k(step);
        // processor subinterval which the current rank belongs to
        int partition_idx = P.partition_index(div, rank);
        Interval newP = P.subinterval(div, partition_idx);
        // intervals of M, N and K that the current rank is in charge of,
        // together with other ranks from its group.
        // (see the definition of group and offset below)
        Interval newm = m.subinterval(divm, divm>1 ? partition_idx : 0);
        Interval newn = n.subinterval(divn, divn>1 ? partition_idx : 0);
        Interval newk = k.subinterval(divk, divk>1 ? partition_idx : 0); 
        bool expanded = false;

        int offset = rank - newP.first();

        std::vector<std::vector<int>> size_before_expansion(P.length());
        std::vector<int> total_before_expansion(P.length());
        std::vector<std::vector<int>> size_after_expansion(newP.length());
        std::vector<int> total_after_expansion(newP.length());

        if ((label_ == 'A' && !strategy.split_A(step))
                || (label_ == 'B' && !strategy.split_B(step))
                || (label_ == 'C' && !strategy.split_C(step))) {

            expanded = true;

            /*
             * this gives us the 2D interval of the matrix that will be expanded:
                 if divm > 1 => matrix B expanded => Interval2D(k, n)
                 if divn > 1 => matrix A expanded => Interval2D(m, k)
                 if divk > 1 => matrix C expanded => Interval2D(m, n)
            */
            Interval2D range;

            if (divm > 1)
                range = Interval2D(k, n);
            else if (divn > 1)
                range = Interval2D(m, k);
            else
                range = Interval2D(m, n);

            buffers_before_expansion(P, range,
                size_before_expansion, total_before_expansion);

            buffers_after_expansion(P, newP,
                size_before_expansion, total_before_expansion,
                size_after_expansion, total_after_expansion);

            // increase the buffer sizes before the recursive call
            set_sizes(newP, size_after_expansion);

            // this is the sum of sizes of all the buckets after expansion
            // that the current rank will own.
            // which is also the size of the matrix after expansion
            long long old_size = total_before_expansion[rank - P.first()];
            long long new_size = total_after_expansion[rank - newP.first()];
            long long max_size = std::max(old_size, new_size);
            if (max_size > max_recv_buffer_size_) {
                max_send_buffer_size_ = max_recv_buffer_size_;
                max_recv_buffer_size_ = max_size;
            } else if (max_size > max_send_buffer_size_) {
                max_send_buffer_size_ = max_size;
            }
        }

        // invoke the recursion
        compute_max_buffer_size(newm, newn, newk, newP, step+1, strategy, rank);

        if (expanded) {
            // the buffer sizes are back to the previous values
            // (the values at the beginning of this BFS step)
            set_sizes(newP, size_before_expansion, newP.first() - P.first());
        }
    }

    //unshift(offset);
    set_dfs_buckets(P, buckets);
}

const long long CarmaMatrix::max_send_buffer_size() const {
    return max_send_buffer_size_;
}
const long long CarmaMatrix::max_recv_buffer_size() const {
    return max_recv_buffer_size_;
}

const std::vector<Interval2D>& CarmaMatrix::initial_layout(int rank) const {
    return mapper_->initial_layout(rank);
}

const std::vector<Interval2D>& CarmaMatrix::initial_layout() const {
    return initial_layout();
}

// (gi, gj) -> (local_id, rank)
std::pair<int, int> CarmaMatrix::local_coordinates(int gi, int gj) {
    return mapper_->local_coordinates(gi, gj);
}

// (local_id, rank) -> (gi, gj)
std::pair<int, int> CarmaMatrix::global_coordinates(int local_index, int rank) {
    return mapper_->global_coordinates(local_index, rank);
}

// local_id -> (gi, gj) for local elements on the current rank
const std::pair<int, int> CarmaMatrix::global_coordinates(int local_index) const {
    return mapper_->global_coordinates(local_index);
}

double* CarmaMatrix::matrix_pointer() {
    return matrix_.data();
}

std::vector<double>& CarmaMatrix::matrix() {
    return matrix_;
}

char CarmaMatrix::which_matrix() {
    return label_;
}

int CarmaMatrix::shift(int rank, int dfs_bucket) {
    int offset = layout_->offset(rank, dfs_bucket);
    current_mat += offset;
    return offset;
}

int CarmaMatrix::shift(int dfs_bucket) {
    int offset = layout_->offset(dfs_bucket);
    current_mat += offset;
    return offset;
}


void CarmaMatrix::unshift(int offset) {
    current_mat -= offset;
}

int CarmaMatrix::dfs_bucket(int rank) {
    return layout_->dfs_bucket(rank); 
}

int CarmaMatrix::dfs_bucket() {
    return layout_->dfs_bucket();
}

void CarmaMatrix::update_buckets(Interval& P, Interval2D& range) {
    layout_->update_buckets(P, range);
}

std::vector<int> CarmaMatrix::dfs_buckets(Interval& newP) {
    return layout_->dfs_buckets(newP);
}

void CarmaMatrix::set_dfs_buckets(Interval& newP, std::vector<int>& pointers) {
    layout_->set_dfs_buckets(newP, pointers);
}

int CarmaMatrix::size(int rank) {
    return layout_->size(rank);
}

int CarmaMatrix::size() {
    return layout_->size();
}

void CarmaMatrix::buffers_before_expansion(Interval& P, Interval2D& range,
        std::vector<std::vector<int>>& size_per_rank,
        std::vector<int>& total_size_per_rank) {
    layout_->buffers_before_expansion(P, range, size_per_rank, total_size_per_rank);
}

void CarmaMatrix::buffers_after_expansion(Interval& P, Interval& newP,
        std::vector<std::vector<int>>& size_per_rank,
        std::vector<int>& total_size_per_rank,
        std::vector<std::vector<int>>& new_size,
        std::vector<int>& new_total) {
    layout_->buffers_after_expansion(P, newP, size_per_rank, total_size_per_rank,
            new_size, new_total);
}

void CarmaMatrix::set_sizes(Interval& newP, std::vector<std::vector<int>>& size_per_rank, int offset) {
    layout_->set_sizes(newP, size_per_rank, offset);
}
void CarmaMatrix::set_sizes(Interval& newP, std::vector<std::vector<int>>& size_per_rank) {
    layout_->set_sizes(newP, size_per_rank);
}

void CarmaMatrix::set_sizes(int rank, std::vector<int>& sizes, int start) {
    layout_->set_sizes(rank, sizes, start);
}

double& CarmaMatrix::operator[](const std::vector<double>::size_type index) {
    return matrix()[index];
}

std::ostream& operator<<(std::ostream& os, const CarmaMatrix& mat) {
    for (auto local = 0; local < mat.initial_size(); ++local) {
        double value = mat.matrix_[local];
        int row, col;
        std::tie(row, col) = mat.global_coordinates(local);
        os << row << " " << col << " " << value << std::endl; 
    }
    return os;
}

void CarmaMatrix::load_data() {
    std::copy(matrix_.begin(), matrix_.end(), send_buffer_.begin());
}

void CarmaMatrix::unload_data() {
    std::copy(send_buffer_.begin(), send_buffer_.begin() + initial_size(), matrix_.begin());
}

double* CarmaMatrix::send_buffer() {
    return send_buffer_.data();
}

double* CarmaMatrix::receive_buffer() {
    return receive_buffer_.data();
}

double* CarmaMatrix::current_matrix() {
    return current_mat;
}

void CarmaMatrix::set_current_matrix(double* mat) {
    current_mat = mat;
}

void CarmaMatrix::swap_buffers() {
    send_buffer_.swap(receive_buffer_);
    //current_mat = send_buffer_.data();
}
