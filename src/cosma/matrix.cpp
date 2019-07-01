#include <cosma/matrix.hpp>

#include <complex>

namespace cosma {

extern template class Buffer<double>;

template <typename T>
CosmaMatrix<T>::CosmaMatrix(char label,
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

template <typename T>
int CosmaMatrix<T>::m() {
    return m_;
}

template <typename T>
int CosmaMatrix<T>::n() {
    return n_;
}

template <typename T>
char CosmaMatrix<T>::label() {
    return label_;
}

template <typename T>
int CosmaMatrix<T>::initial_size(int rank) const {
    if (rank >= strategy_.P)
        return 0;
    return mapper_.initial_size(rank);
}

template <typename T>
int CosmaMatrix<T>::initial_size() const {
    if (rank_ >= strategy_.P)
        return 0;
    return mapper_.initial_size();
}

template <typename T>
int CosmaMatrix<T>::buffer_index() {
    return buffer_.buffer_index();
}

template <typename T>
void CosmaMatrix<T>::set_buffer_index(int idx) {
    buffer_.set_buffer_index(idx);
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::buffer_ptr() {
    return buffer_.buffer_ptr();
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::reshuffle_buffer_ptr() {
    return buffer_.reshuffle_buffer_ptr();
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::reduce_buffer_ptr() {
    return buffer_.reduce_buffer_ptr();
}

template <typename T>
typename CosmaMatrix<T>::mpi_buffer_t &CosmaMatrix<T>::buffer() {
    return buffer_.buffer();
}

template <typename T>
const typename CosmaMatrix<T>::mpi_buffer_t &CosmaMatrix<T>::buffer() const {
    return buffer_.buffer();
}

template <typename T>
void CosmaMatrix<T>::advance_buffer() {
    buffer_.advance_buffer();
}

template <typename T>
const std::vector<Interval2D> &CosmaMatrix<T>::initial_layout(int rank) const {
    return mapper_.initial_layout(rank);
}

template <typename T>
const std::vector<Interval2D> &CosmaMatrix<T>::initial_layout() const {
    return initial_layout();
}

// (gi, gj) -> (local_id, rank)
template <typename T>
std::pair<int, int> CosmaMatrix<T>::local_coordinates(int gi, int gj) {
    return mapper_.local_coordinates(gi, gj);
}

// (local_id, rank) -> (gi, gj)
template <typename T>
std::pair<int, int> CosmaMatrix<T>::global_coordinates(int local_index,
                                                       int rank) {
    return mapper_.global_coordinates(local_index, rank);
}

// local_id -> (gi, gj) for local elements on the current rank
template <typename T>
const std::pair<int, int>
CosmaMatrix<T>::global_coordinates(int local_index) const {
    return mapper_.global_coordinates(local_index);
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::matrix_pointer() {
    // return matrix_.data();
    if (rank_ >= strategy_.P) {
        return nullptr;
    }
    return buffer_.initial_buffer_ptr();
}

template <typename T>
typename CosmaMatrix<T>::mpi_buffer_t &CosmaMatrix<T>::matrix() {
    // return matrix_;
    if (rank_ >= strategy_.P) {
        return dummy_vector;
    }
    return buffer_.initial_buffer();
}

template <typename T>
const typename CosmaMatrix<T>::mpi_buffer_t &CosmaMatrix<T>::matrix() const {
    // return matrix_;
    if (rank_ >= strategy_.P) {
        return dummy_vector;
    }
    return buffer_.initial_buffer();
}

template <typename T>
char CosmaMatrix<T>::which_matrix() {
    return label_;
}

template <typename T>
int CosmaMatrix<T>::shift(int rank, int seq_bucket) {
    int offset = layout_.offset(rank, seq_bucket);
    current_mat += offset;
    return offset;
}

template <typename T>
int CosmaMatrix<T>::shift(int seq_bucket) {
    int offset = layout_.offset(seq_bucket);
    current_mat += offset;
    return offset;
}

template <typename T>
void CosmaMatrix<T>::unshift(int offset) {
    current_mat -= offset;
}

template <typename T>
int CosmaMatrix<T>::seq_bucket(int rank) {
    return layout_.seq_bucket(rank);
}

template <typename T>
int CosmaMatrix<T>::seq_bucket() {
    return layout_.seq_bucket();
}

template <typename T>
void CosmaMatrix<T>::update_buckets(Interval &P, Interval2D &range) {
    layout_.update_buckets(P, range);
}

template <typename T>
std::vector<int> CosmaMatrix<T>::seq_buckets(Interval &newP) {
    return layout_.seq_buckets(newP);
}

template <typename T>
void CosmaMatrix<T>::set_seq_buckets(Interval &newP,
                                     std::vector<int> &pointers) {
    layout_.set_seq_buckets(newP, pointers);
}

template <typename T>
int CosmaMatrix<T>::size(int rank) {
    return layout_.size(rank);
}

template <typename T>
int CosmaMatrix<T>::size() {
    return layout_.size();
}

template <typename T>
void CosmaMatrix<T>::buffers_before_expansion(
    Interval &P,
    Interval2D &range,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank) {
    layout_.buffers_before_expansion(
        P, range, size_per_rank, total_size_per_rank);
}

template <typename T>
void CosmaMatrix<T>::buffers_after_expansion(
    Interval &P,
    Interval &newP,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank,
    std::vector<std::vector<int>> &new_size,
    std::vector<int> &new_total) {
    layout_.buffers_after_expansion(
        P, newP, size_per_rank, total_size_per_rank, new_size, new_total);
}

template <typename T>
void CosmaMatrix<T>::set_sizes(Interval &newP,
                               std::vector<std::vector<int>> &size_per_rank,
                               int offset) {
    layout_.set_sizes(newP, size_per_rank, offset);
}

template <typename T>
void CosmaMatrix<T>::set_sizes(Interval &newP,
                               std::vector<std::vector<int>> &size_per_rank) {
    layout_.set_sizes(newP, size_per_rank);
}

template <typename T>
void CosmaMatrix<T>::set_sizes(int rank, std::vector<int> &sizes, int start) {
    layout_.set_sizes(rank, sizes, start);
}

template <typename T>
typename CosmaMatrix<T>::scalar_t &CosmaMatrix<T>::
operator[](const typename std::vector<scalar_t>::size_type index) {
    return matrix()[index];
}

template <typename T>
typename CosmaMatrix<T>::scalar_t CosmaMatrix<T>::
operator[](const typename std::vector<scalar_t>::size_type index) const {
    return matrix()[index];
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::current_matrix() {
    return current_mat;
}

template <typename T>
void CosmaMatrix<T>::set_current_matrix(scalar_t *mat) {
    current_mat = mat;
}

// Explicit instantiations
//
template class CosmaMatrix<float>;
template class CosmaMatrix<double>;
template class CosmaMatrix<std::complex<float>>;
template class CosmaMatrix<std::complex<double>>;

} // namespace cosma
