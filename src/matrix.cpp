#include "matrix.hpp"

CarmaMatrix::CarmaMatrix(char label, const Strategy& strategy, int rank) :
        label_(label), rank_(rank), strategy_(strategy) {
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

    compute_n_buckets();

    initialize_buffers();

    //matrix_ = std::vector<double>(mapper_->initial_size(rank));

    current_mat = matrix_pointer();

    PL();
}

int CarmaMatrix::m() {
    return m_;
}

int CarmaMatrix::n() {
    return n_;
}

char CarmaMatrix::label() {
    return label_;
}

const int CarmaMatrix::initial_size(int rank) const {
    return mapper_->initial_size(rank);
}

const int CarmaMatrix::initial_size() const {
    return mapper_->initial_size();
}

void CarmaMatrix::compute_n_buckets() {
    n_buckets_ = std::vector<int>(strategy_.n_steps);
    expanded_after_ = std::vector<bool>(strategy_.n_steps);
    int prod_n_dfs = 1;

    bool expanded = false;

    for (int step = strategy_.n_steps - 1; step >= 0; --step) {
        n_buckets_[step] = prod_n_dfs;
        // if the current step is DFS and this matrix was split
        // then update the product of all dfs steps in 
        // which this matrix was split, which represents
        // the number of buckets
        if (strategy_.dfs_step(step)) {
            if ((label_ == 'A' && strategy_.split_A(step))
                || (label_ == 'B' && strategy_.split_B(step))
                || (label_ == 'C' && strategy_.split_C(step))) {
                prod_n_dfs *= strategy_.divisor(step);
            }
        } else {
            // if the current matrix was expanded (i.e. NOT split)
            if ((label_ == 'A' && !strategy_.split_A(step))
                || (label_ == 'B' && !strategy_.split_B(step))
                || (label_ == 'C' && !strategy_.split_C(step))) {
                expanded = true;
            }
        }
        expanded_after_[step] = expanded;
    }
}

void CarmaMatrix::initialize_buffers() {
    max_send_buffer_size_ = (long long) initial_size();
    max_recv_buffer_size_ = (long long) initial_size();
    std::vector<long long> buff_sizes = compute_buffer_size(strategy_);

    buffers_ = std::vector<std::vector<double>>(buff_sizes.size()+1, std::vector<double>());
    buffers_[0] = std::vector<double>(initial_size());
    // ignore the first buffer size since it's already allocated
    // in the initial buffers
    for (int i = 0; i < buff_sizes.size(); ++i) {
        buffers_[i+1].resize(buff_sizes[i]);
    }

    current_buffer_ = 0;
}

void CarmaMatrix::advance_buffer() {
    if (current_buffer_ == buffers_.size() - 1)
        current_buffer_--;
    else
        current_buffer_++;

    // should never happen
    if (current_buffer_ < 0)
        current_buffer_ = 0;
}

int CarmaMatrix::buffer_index() {
    return current_buffer_;
}

void CarmaMatrix::set_buffer_index(int idx) {
    current_buffer_ = idx;
}

double* CarmaMatrix::receiving_buffer() {
    return buffers_[current_buffer_].data();
}

std::vector<long long> CarmaMatrix::compute_buffer_size(const Strategy& strategy) {
    Interval m(0, strategy.m - 1);
    Interval n(0, strategy.n - 1);
    Interval k(0, strategy.k - 1);
    Interval P(0, strategy.P - 1);

    return compute_buffer_size(m, n, k, P, 0, strategy, rank_);
}

std::vector<long long> CarmaMatrix::compute_buffer_size(Interval& m, Interval& n, Interval& k, 
    Interval& P, int step, const Strategy& strategy, int rank) {
    if (strategy.final_step(step)) return {};
    std::vector<long long> sizes;
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


    // recursively invoke BFS or DFS:
    if (n_buckets_[step] == 1) {
        compute_max_buffer_size(m, n, k, P, step, strategy, rank);
        if (expanded_after_[step])
            return {max_send_buffer_size_, max_recv_buffer_size_};
        else
            return {max_recv_buffer_size_};
    }
    // recursively invoke BFS or DFS:
    if (strategy.dfs_step(step)) {
        int div = strategy.divisor(step);
        int divm = strategy.divisor_m(step);
        int divn = strategy.divisor_n(step);
        int divk = strategy.divisor_k(step);

        for (int i = 0; i < div; ++i) {
            Interval newm = m.subinterval(divm, divm>1 ? i : 0);
            Interval newn = n.subinterval(divn, divn>1 ? i : 0);
            Interval newk = k.subinterval(divk, divk>1 ? i : 0);

            std::vector<long long> subsizes = compute_buffer_size(newm, newn, newk, P, 
                    step+1, strategy, rank);

            if (i == 0) {
                sizes = std::vector<long long>(subsizes.size());
            }

            //sizes[0] += subsizes[0];

            for (int j = 0; j < sizes.size(); ++j) {
                sizes[j] = std::max(sizes[j], subsizes[j]);
            }

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

        long long max_size = -1;

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
            max_size = std::max(old_size, new_size);
        }

        // invoke the recursion
        std::vector<long long> subsizes = compute_buffer_size(newm, newn, newk, newP, 
                step+1, strategy, rank);

        if (expanded) {
            sizes = std::vector<long long>(subsizes.size() + 1);
            sizes[0] = max_size;
            std::copy(subsizes.begin(), subsizes.end(), sizes.begin() + 1);
            // the buffer sizes are back to the previous values
            // (the values at the beginning of this BFS step)
            set_sizes(newP, size_before_expansion, newP.first() - P.first());
        } else {
            sizes = subsizes;
        }
    }

    //unshift(offset);
    set_dfs_buckets(P, buckets);
    return sizes;
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
    //return matrix_.data();
    return buffers_[0].data();
}

std::vector<double>& CarmaMatrix::matrix() {
    //return matrix_;
    return buffers_[0];
}

const std::vector<double>& CarmaMatrix::matrix() const {
    //return matrix_;
    return buffers_[0];
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

double CarmaMatrix::operator[](const std::vector<double>::size_type index) const {
    return matrix()[index];
}

std::ostream& operator<<(std::ostream& os, const CarmaMatrix& mat) {
    for (auto local = 0; local < mat.initial_size(); ++local) {
        double value = mat[local];
        int row, col;
        std::tie(row, col) = mat.global_coordinates(local);
        os << row << " " << col << " " << value << std::endl; 
    }
    return os;
}

double* CarmaMatrix::current_matrix() {
    return current_mat;
}

void CarmaMatrix::set_current_matrix(double* mat) {
    current_mat = mat;
}

