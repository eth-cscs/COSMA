#include <cosma/bfloat16.hpp>
#include <cosma/matrix.hpp>
#include <mpi.h>

#include <complex>

namespace cosma {

extern template class Buffer<double>;

// using a pointer to cosma_context
template <typename T>
CosmaMatrix<T>::CosmaMatrix(cosma_context<T> *ctxt,
                            char label,
                            const Strategy &strategy,
                            int rank,
                            bool dry_run)
    : ctxt_(ctxt)
    , mapper_(Mapper(label, strategy, rank))
    , rank_(mapper_.rank())
    , strategy_(mapper_.strategy())
    , label_(mapper_.label())
    , m_(mapper_.m())
    , n_(mapper_.n())
    , P_(mapper_.P()) {

    if (rank < P_) {
        layout_ = Layout(&mapper_);

        buffer_ = buffer_t(ctxt_, &mapper_, &layout_, dry_run);
    }
}

// with given mapper
template <typename T>
CosmaMatrix<T>::CosmaMatrix(cosma_context<T> *ctxt,
                            Mapper &&mapper,
                            int rank,
                            bool dry_run)
    : ctxt_(ctxt)
    , mapper_(std::forward<Mapper>(mapper))
    , rank_(rank)
    , strategy_(mapper_.strategy())
    , label_(mapper_.label())
    , m_(mapper_.m())
    , n_(mapper_.n())
    , P_(mapper_.P()) {

    mapper_.reorder_rank(rank);
    if (rank < P_) {
        layout_ = Layout(&mapper_);
        buffer_ = buffer_t(ctxt_, &mapper_, &layout_, dry_run);
    }
}

// using custom context
template <typename T>
CosmaMatrix<T>::CosmaMatrix(std::unique_ptr<cosma_context<T>> &ctxt,
                            char label,
                            const Strategy &strategy,
                            int rank,
                            bool dry_run)
    : CosmaMatrix(ctxt.get(), label, strategy, rank, dry_run) {}

// with given mapper
template <typename T>
CosmaMatrix<T>::CosmaMatrix(std::unique_ptr<cosma_context<T>> &ctxt,
                            Mapper &&mapper,
                            int rank,
                            bool dry_run)
    : CosmaMatrix(ctxt.get(), std::forward<Mapper &&>(mapper), rank, dry_run) {}

// using global (singleton) context
template <typename T>
CosmaMatrix<T>::CosmaMatrix(char label,
                            const Strategy &strategy,
                            int rank,
                            bool dry_run)
    : CosmaMatrix(get_context_instance<T>(), label, strategy, rank, dry_run) {}

// with given mapper
template <typename T>
CosmaMatrix<T>::CosmaMatrix(Mapper &&mapper, int rank, bool dry_run)
    : CosmaMatrix(get_context_instance<T>(),
                  std::forward<Mapper &&>(mapper),
                  rank,
                  dry_run) {}

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
int CosmaMatrix<T>::buffer_index() {
    if (rank_ < P_) {
        return buffer_.buffer_index();
    }
    return -1;
}

template <typename T>
void CosmaMatrix<T>::set_buffer_index(int idx) {
    if (rank_ < P_) {
        buffer_.set_buffer_index(idx);
    }
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::buffer_ptr() {
    if (rank_ < P_) {
        return buffer_.buffer_ptr();
    }
    return nullptr;
}

template <typename T>
size_t CosmaMatrix<T>::buffer_size() {
    if (rank_ < P_) {
        return buffer_.buffer_size();
    }
    return 0;
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::reshuffle_buffer_ptr() {
    if (rank_ < P_) {
        return buffer_.reshuffle_buffer_ptr();
    }
    return nullptr;
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::reduce_buffer_ptr() {
    if (rank_ < P_) {
        return buffer_.reduce_buffer_ptr();
    }
    return nullptr;
}

template <typename T>
void CosmaMatrix<T>::swap_reduce_buffer_with(size_t buffer_idx) {
    if (rank_ < P_) {
        buffer_.swap_reduce_buffer_with(buffer_idx);
    }
}

template <typename T>
void CosmaMatrix<T>::advance_buffer() {
    if (rank_ < P_) {
        buffer_.advance_buffer();
    }
}

template <typename T>
const std::vector<Interval2D> &CosmaMatrix<T>::initial_layout(int rank) const {
    return mapper_.initial_layout(rank);
}

template <typename T>
const std::vector<Interval2D> &CosmaMatrix<T>::initial_layout() const {
    return mapper_.initial_layout();
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
std::pair<int, int> CosmaMatrix<T>::global_coordinates(int local_index) {
    return mapper_.global_coordinates(local_index);
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::matrix_pointer() {
    if (rank_ < P_) {
        return buffer_.initial_buffer_ptr();
    }
    return nullptr;
}

template <typename T>
const typename CosmaMatrix<T>::scalar_t *
CosmaMatrix<T>::matrix_pointer() const {
    if (rank_ < P_) {
        return buffer_.initial_buffer_ptr();
    }
    return nullptr;
}

template <typename T>
size_t CosmaMatrix<T>::matrix_size() const {
    return mapper_.initial_size();
}

template <typename T>
size_t CosmaMatrix<T>::matrix_size(int rank) const {
    return mapper_.initial_size(rank);
}

template <typename T>
char CosmaMatrix<T>::which_matrix() {
    return label_;
}

template <typename T>
int CosmaMatrix<T>::shift(int rank, int seq_bucket) {
    if (rank < P_) {
        int offset = layout_.offset(rank, seq_bucket);
        current_mat += offset;
        return offset;
    }
    return -1;
}

template <typename T>
int CosmaMatrix<T>::shift(int seq_bucket) {
    if (rank_ < P_) {
        int offset = layout_.offset(seq_bucket);
        current_mat += offset;
        return offset;
    }
    return -1;
}

template <typename T>
void CosmaMatrix<T>::unshift(int offset) {
    if (rank_ < P_) {
        current_mat -= offset;
    }
}

template <typename T>
int CosmaMatrix<T>::seq_bucket(int rank) {
    if (rank < P_) {
        return layout_.seq_bucket(rank);
    }
    return -1;
}

template <typename T>
int CosmaMatrix<T>::seq_bucket() {
    if (rank_ < P_) {
        return layout_.seq_bucket();
    }
    return -1;
}

template <typename T>
void CosmaMatrix<T>::update_buckets(Interval &P, Interval2D &range) {
    if (rank_ < P_) {
        layout_.update_buckets(P, range);
    }
}

template <typename T>
std::vector<int> CosmaMatrix<T>::seq_buckets(Interval &newP) {
    if (rank_ < P_) {
        return layout_.seq_buckets(newP);
    }
    return {};
}

template <typename T>
void CosmaMatrix<T>::set_seq_buckets(Interval &newP,
                                     std::vector<int> &pointers) {
    if (rank_ < P_) {
        layout_.set_seq_buckets(newP, pointers);
    }
}

template <typename T>
int CosmaMatrix<T>::size(int rank) {
    if (rank < P_) {
        return layout_.size(rank);
    }
    return 0;
}

template <typename T>
int CosmaMatrix<T>::size() {
    if (rank_ < P_) {
        return layout_.size();
    }
    return 0;
}

template <typename T>
void CosmaMatrix<T>::buffers_before_expansion(
    Interval &P,
    Interval2D &range,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank) {
    if (rank_ < P_) {
        layout_.buffers_before_expansion(
            P, range, size_per_rank, total_size_per_rank);
    }
}

template <typename T>
void CosmaMatrix<T>::buffers_after_expansion(
    Interval &P,
    Interval &newP,
    std::vector<std::vector<int>> &size_per_rank,
    std::vector<int> &total_size_per_rank,
    std::vector<std::vector<int>> &new_size,
    std::vector<int> &new_total) {
    if (rank_ < P_) {
        layout_.buffers_after_expansion(
            P, newP, size_per_rank, total_size_per_rank, new_size, new_total);
    }
}

template <typename T>
void CosmaMatrix<T>::set_sizes(Interval &newP,
                               std::vector<std::vector<int>> &size_per_rank,
                               int offset) {
    if (rank_ < P_) {
        layout_.set_sizes(newP, size_per_rank, offset);
    }
}

template <typename T>
void CosmaMatrix<T>::set_sizes(Interval &newP,
                               std::vector<std::vector<int>> &size_per_rank) {
    if (rank_ < P_) {
        layout_.set_sizes(newP, size_per_rank);
    }
}

template <typename T>
void CosmaMatrix<T>::set_sizes(int rank, std::vector<int> &sizes, int start) {
    if (rank < P_) {
        layout_.set_sizes(rank, sizes, start);
    }
}

template <typename T>
typename CosmaMatrix<T>::scalar_t &CosmaMatrix<T>::operator[](
    const typename std::vector<scalar_t>::size_type index) {
    if (index < matrix_size()) {
        std::runtime_error("Matrix index out of bounds.");
    }
    return matrix_pointer()[index];
}

template <typename T>
typename CosmaMatrix<T>::scalar_t CosmaMatrix<T>::operator[](
    const typename std::vector<scalar_t>::size_type index) const {
    if (index < matrix_size()) {
        std::runtime_error("Matrix index out of bounds.");
    }
    return matrix_pointer()[index];
}

template <typename T>
typename CosmaMatrix<T>::scalar_t *CosmaMatrix<T>::current_matrix() {
    return current_mat;
}

template <typename T>
void CosmaMatrix<T>::initialize() {
    current_mat = matrix_pointer();
}

template <typename T>
void CosmaMatrix<T>::set_current_matrix(scalar_t *mat) {
    current_mat = mat;
}

template <typename T>
costa::grid_layout<T> CosmaMatrix<T>::get_grid_layout() {
    // **************************
    // get an assigned grid2D
    // **************************
    auto assigned_grid = mapper_.get_layout_grid();

    // **************************
    // create local memory view
    // **************************
    // get coordinates of current rank in a rank decomposition
    std::vector<costa::block<T>> loc_blocks;
    for (auto matrix_id = 0u; matrix_id < mapper_.local_blocks().size();
         ++matrix_id) {
        Interval2D range = mapper_.local_blocks()[matrix_id];
        int offset = mapper_.local_blocks_offsets()[matrix_id];

        costa::interval row_interval(range.rows.first(), range.rows.last() + 1);
        costa::interval col_interval(range.cols.first(), range.cols.last() + 1);

        int stride = row_interval.length();

        costa::block<T> b(assigned_grid,
                          row_interval,
                          col_interval,
                          matrix_pointer() + offset,
                          stride);

        assert(b.non_empty());

        loc_blocks.push_back(b);
    }
    costa::local_blocks<T> local_memory(std::move(loc_blocks));

    return {std::move(assigned_grid), std::move(local_memory), 'C'};
}

// allocates initial buffers (turns off dryrun)
template <typename T>
void CosmaMatrix<T>::allocate() {
    if (rank_ < P_) {
        bool dryrun = false;
        buffer_.allocate_initial_buffers(dryrun);
        initialize();
    }
}

template <typename T>
void CosmaMatrix<T>::allocate_communication_buffers() {
    if (rank_ < P_)
        buffer_.allocate_communication_buffers();
}

template <typename T>
void CosmaMatrix<T>::free_communication_buffers() {
    if (rank_ < P_)
        buffer_.free_communication_buffers();
}

template <typename T>
cosma_context<T> *CosmaMatrix<T>::get_context() {
    return ctxt_;
}

template <typename T>
int CosmaMatrix<T>::rank() const {
    return rank_;
}

// total memory = initial memory + communication memory
template <typename T>
std::vector<size_t> CosmaMatrix<T>::required_memory() {
    if (rank_ < P_)
        return buffer_.get_all_buffer_sizes();
    return std::vector<std::size_t>{};
}

// Explicit instantiations
//
template class CosmaMatrix<float>;
template class CosmaMatrix<double>;
template class CosmaMatrix<std::complex<float>>;
template class CosmaMatrix<std::complex<double>>;
template class CosmaMatrix<bfloat16>;

} // namespace cosma
