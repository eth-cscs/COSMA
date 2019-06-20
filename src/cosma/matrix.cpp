#include <cosma/matrix.hpp>

namespace cosma {

extern template class Buffer<double>;

CosmaMatrix::CosmaMatrix(char label,
                         const Strategy &strategy,
                         int rank,
                         bool dry_run)
    : label_(label)
    , rank_(rank)
    , strategy_(strategy) {
    PE(preprocessing_matrices);
    if (label_ == 'A') {
        m_ = strategy.m;
        n_ = strategy.k;
    } else if (label_ == 'B') {
        m_ = strategy.k;
        n_ = strategy.n;
    } else {
        m_ = strategy.m;
        n_ = strategy.n;
    }
    P_ = strategy.P;

    if (rank >= P_) {
        return;
    }

    mapper_ = Mapper(label, m_, n_, P_, strategy, rank);
    layout_ = Layout(label, m_, n_, P_, rank, mapper_.complete_layout());
    buffer_ = buffer_t(label, strategy, rank, &mapper_, &layout_, dry_run);

    current_mat = matrix_pointer();

    PL();
}

int CosmaMatrix::m() { return m_; }

int CosmaMatrix::n() { return n_; }

char CosmaMatrix::label() { return label_; }

int CosmaMatrix::initial_size(int rank) const {
    if (rank >= strategy_.P)
        return 0;
    return mapper_.initial_size(rank);
}

int CosmaMatrix::initial_size() const {
    if (rank_ >= strategy_.P)
        return 0;
    return mapper_.initial_size();
}

int CosmaMatrix::buffer_index() { return buffer_.buffer_index(); }

void CosmaMatrix::set_buffer_index(int idx) { buffer_.set_buffer_index(idx); }

double *CosmaMatrix::buffer_ptr() { return buffer_.buffer_ptr(); }

double *CosmaMatrix::reshuffle_buffer_ptr() {
    return buffer_.reshuffle_buffer_ptr();
}

double *CosmaMatrix::reduce_buffer_ptr() { return buffer_.reduce_buffer_ptr(); }

mpi_buffer_t &CosmaMatrix::buffer() { return buffer_.buffer(); }

const mpi_buffer_t &CosmaMatrix::buffer() const { return buffer_.buffer(); }

void CosmaMatrix::advance_buffer() { buffer_.advance_buffer(); }

const std::vector<Interval2D> &CosmaMatrix::initial_layout(int rank) const {
    return mapper_.initial_layout(rank);
}

const std::vector<Interval2D> &CosmaMatrix::initial_layout() const {
    return initial_layout();
}

// (gi, gj) -> (local_id, rank)
std::pair<int, int> CosmaMatrix::local_coordinates(int gi, int gj) {
    return mapper_.local_coordinates(gi, gj);
}

// (local_id, rank) -> (gi, gj)
std::pair<int, int> CosmaMatrix::global_coordinates(int local_index, int rank) {
    return mapper_.global_coordinates(local_index, rank);
}

// local_id -> (gi, gj) for local elements on the current rank
const std::pair<int, int>
CosmaMatrix::global_coordinates(int local_index) const {
    return mapper_.global_coordinates(local_index);
}

double *CosmaMatrix::matrix_pointer() {
    // return matrix_.data();
    if (rank_ >= strategy_.P) {
        return nullptr;
    }
    return buffer_.initial_buffer_ptr();
}

mpi_buffer_t &CosmaMatrix::matrix() {
    // return matrix_;
    if (rank_ >= strategy_.P) {
        return dummy_vector;
    }
    return buffer_.initial_buffer();
}

const mpi_buffer_t &CosmaMatrix::matrix() const {
    // return matrix_;
    if (rank_ >= strategy_.P) {
        return dummy_vector;
    }
    return buffer_.initial_buffer();
}

char CosmaMatrix::which_matrix() { return label_; }

int CosmaMatrix::shift(int rank, int seq_bucket) {
    int offset = layout_.offset(rank, seq_bucket);
    current_mat += offset;
    return offset;
}

int CosmaMatrix::shift(int seq_bucket) {
    int offset = layout_.offset(seq_bucket);
    current_mat += offset;
    return offset;
}

void CosmaMatrix::unshift(int offset) { current_mat -= offset; }

int CosmaMatrix::seq_bucket(int rank) { return layout_.seq_bucket(rank); }

int CosmaMatrix::seq_bucket() { return layout_.seq_bucket(); }

void CosmaMatrix::update_buckets(Interval &P, Interval2D &range) {
    layout_.update_buckets(P, range);
}

std::vector<int> CosmaMatrix::seq_buckets(Interval &newP) {
    return layout_.seq_buckets(newP);
}

void CosmaMatrix::set_seq_buckets(Interval &newP, std::vector<int> &pointers) {
    layout_.set_seq_buckets(newP, pointers);
}

int CosmaMatrix::size(int rank) { return layout_.size(rank); }

int CosmaMatrix::size() { return layout_.size(); }

void CosmaMatrix::buffers_before_expansion(
    Interval &P,
    Interval2D &range,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank) {
    layout_.buffers_before_expansion(
        P, range, size_per_rank, total_size_per_rank);
}

void CosmaMatrix::buffers_after_expansion(
    Interval &P,
    Interval &newP,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank,
    std::vector<std::vector<int>> &new_size,
    std::vector<int> &new_total) {
    layout_.buffers_after_expansion(
        P, newP, size_per_rank, total_size_per_rank, new_size, new_total);
}

void CosmaMatrix::set_sizes(Interval &newP,
                            std::vector<std::vector<int>> &size_per_rank,
                            int offset) {
    layout_.set_sizes(newP, size_per_rank, offset);
}
void CosmaMatrix::set_sizes(Interval &newP,
                            std::vector<std::vector<int>> &size_per_rank) {
    layout_.set_sizes(newP, size_per_rank);
}

void CosmaMatrix::set_sizes(int rank, std::vector<int> &sizes, int start) {
    layout_.set_sizes(rank, sizes, start);
}

double &CosmaMatrix::operator[](const std::vector<double>::size_type index) {
    return matrix()[index];
}

double CosmaMatrix::
operator[](const std::vector<double>::size_type index) const {
    return matrix()[index];
}

std::ostream &operator<<(std::ostream &os, const CosmaMatrix &mat) {
    for (auto local = 0; local < mat.initial_size(); ++local) {
        double value = mat[local];
        int row, col;
        std::tie(row, col) = mat.global_coordinates(local);
        os << row << " " << col << " " << value << std::endl;
    }
    return os;
}

double *CosmaMatrix::current_matrix() { return current_mat; }

void CosmaMatrix::set_current_matrix(double *mat) { current_mat = mat; }
} // namespace cosma
