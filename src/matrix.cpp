#include "matrix.hpp"

CarmaMatrix::CarmaMatrix(char label, const Strategy& strategy, int rank) :
        label_(label), rank_(rank) {
    PE(preprocessing);
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
    current_mat = matrix_.data();
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

double* CarmaMatrix::current_matrix() {
    return current_mat;
}

void CarmaMatrix::set_current_matrix(double* mat) {
    current_mat = mat;
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
