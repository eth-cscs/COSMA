#include <complex>
#include <cosma/bfloat16.hpp>
#include <cosma/buffer.hpp>
#include <cosma/context.hpp>
#include <cosma/profiler.hpp>

#include <algorithm>

namespace cosma {

template <typename T>
Buffer<T>::Buffer()
    : ctxt_(nullptr) {}

template <typename T>
Buffer<T>::Buffer(cosma_context<T> *ctxt,
                  Mapper *mapper,
                  Layout *layout,
                  bool dry_run)
    : ctxt_(ctxt)
    , strategy_(&(mapper->strategy()))
    , label_(mapper->label())
    , rank_(mapper->rank())
    , mapper_(mapper)
    , layout_(layout) {

    PE(preprocessing_matrices_buffer);
    compute_n_buckets();

    max_base_buffer_size_ = 0;
    max_reduce_buffer_size_ = 0;
    max_reshuffle_buffer_size_ = 0;
    current_buffer_ = 0;
    max_send_buffer_size_ = (size_t)mapper_->initial_size();
    max_recv_buffer_size_ = (size_t)mapper_->initial_size();

    init_first_split_steps();
    buff_sizes_ = compute_buffer_size();

    // to account for possible swapping with the reduce buffer
    // that occurs if k split is present and beta != 0
    if (label_ == 'C') {
        for (int step = 0; step < strategy_->n_steps(); ++step) {
            if (strategy_->split_k(step) && strategy_->parallel_step(step)) {
                max_reduce_buffer_size_ = std::max(
                    max_reduce_buffer_size_,
                    *max_element(buff_sizes_.begin(), buff_sizes_.end()));
                break;
            }
        }
    }

    allocate_initial_buffers(dry_run);
    PL();
}

template <typename T>
Buffer<T>::Buffer(Mapper *mapper, Layout *layout, bool dry_run)
    : Buffer(get_context_instance<T>(), mapper, layout, dry_run) {}

template <typename T>
void Buffer<T>::allocate_communication_buffers(bool dry_run) {
    if (!dry_run && rank_ < strategy_->P && buff_sizes_.size() > 1) {
        // check if the initial buffer is already initialized
        assert(buffers_.size() == 1);
        // initial buffer is already allocated, so start from 1
        for (int i = 1; i < buff_sizes_.size(); ++i) {
            auto id = ctxt_->get_memory_pool().get_buffer_id(buff_sizes_[i]);
            buffers_.push_back(id);
        }

        if (max_reshuffle_buffer_size_ > 0) {
            reshuffle_buffer_ = ctxt_->get_memory_pool().get_buffer_id(
                max_reshuffle_buffer_size_);
        }

        if (max_reduce_buffer_size_ > 0) {
            reduce_buffer_ =
                ctxt_->get_memory_pool().get_buffer_id(max_reduce_buffer_size_);
        }
#ifdef DEBUG
        for (int rank = 0; rank < strategy_->P; ++rank) {
            if (rank_ == rank) {
                std::cout << "Rank " << rank_ << " buffers" << std::endl;
                std::cout << "Buffer sizes for matrix " << label_ << " on rank "
                          << rank_ << std::endl;
                std::cout << "max_reshuffle_buffer_size_ = "
                          << max_reshuffle_buffer_size_ << std::endl;
                std::cout << "max_reduce_buffer_size_ = "
                          << max_reduce_buffer_size_ << std::endl;
                std::cout << "max_send_buffer_size_ = " << max_send_buffer_size_
                          << std::endl;
                std::cout << "max_recv_buffer_size_ = " << max_recv_buffer_size_
                          << std::endl;
                std::cout << "max_base_buffer_size_ = " << max_base_buffer_size_
                          << std::endl;
                for (int i = 0; i < buff_sizes_.size(); ++i) {
                    std::cout << "buffer" << i << " size = " << buff_sizes_[i]
                              << std::endl;
                }
            }
            // MPI_Barrier(MPI_COMM_WORLD);
        }
#endif
    }
}

template <typename T>
std::vector<std::size_t> Buffer<T>::get_all_buffer_sizes() {
    std::vector<std::size_t> buffer_sizes;
    if (rank_ < strategy_->P) {
        if (buff_sizes_.size() >= 1) {
            buffer_sizes.push_back(
                std::max((size_t)buff_sizes_[0], mapper_->initial_size()));
        }
        for (int i = 1; i < buff_sizes_.size(); ++i) {
            buffer_sizes.push_back(buff_sizes_[i]);
        }
        if (max_reduce_buffer_size_ > 0) {
            buffer_sizes.push_back(max_reduce_buffer_size_);
        }
        if (max_reshuffle_buffer_size_ > 0) {
            buffer_sizes.push_back(max_reshuffle_buffer_size_);
        }
    }

    return buffer_sizes;
}

template <typename T>
void Buffer<T>::allocate_initial_buffers(bool dry_run) {
    if (!dry_run && rank_ < strategy_->P && buff_sizes_.size() > 0) {
        // Defensive: avoid double allocation if constructor sequence calls this
        // twice.
        if (buffers_.size() != 0) {
#ifdef COSMA_ENABLE_DOUBLE_ALLOC_LOG
            std::cerr << "[COSMA][warn] allocate_initial_buffers called with "
                         "non-empty buffers_ size="
                      << buffers_.size() << " label=" << label_
                      << " rank=" << rank_ << std::endl;
#endif
            return;
        }
        buffers_.reserve(buff_sizes_.size());
        buff_sizes_[0] =
            std::max((size_t)buff_sizes_[0], mapper_->initial_size());
        auto id = ctxt_->get_memory_pool().get_buffer_id(buff_sizes_[0]);
        // (Original assertion removed in favor of guard above)
        buffers_.push_back(id);
    }
}

template <typename T>
void Buffer<T>::free_initial_buffers(bool dry_run) {
    if (!dry_run && rank_ < strategy_->P && buff_sizes_.size() > 0) {
        // check if all the other buffers were deallocated previously
        // buff_sizes_ is equal to n_buffers throughout the lifetime of the
        // class but buffers_ size is decreased whenever some buffer is freed
        assert(buffers_.size() == 1);

        // deallocate initial buffer (that are storing the matrix)
        auto ptr = ctxt_->get_memory_pool().get_buffer_pointer(buffers_[0]);
        ctxt_->get_memory_pool().free_buffer(ptr, buff_sizes_[0]);
        // remove the pointers pointing to them
        buffers_.pop_back();
        buff_sizes_.pop_back();
    }
}

template <typename T>
void Buffer<T>::free_communication_buffers(bool dry_run) {
    if (dry_run || rank_ >= strategy_->P || buff_sizes_.size() <= 1)
        return;
    // deallocate reshuffle and reduce buffers separately
    if (max_reduce_buffer_size_ > 0) {
        auto ptr = ctxt_->get_memory_pool().get_buffer_pointer(reduce_buffer_);
        ctxt_->get_memory_pool().free_buffer(ptr, max_reduce_buffer_size_);
    }

    if (max_reshuffle_buffer_size_ > 0) {
        auto ptr =
            ctxt_->get_memory_pool().get_buffer_pointer(reshuffle_buffer_);
        ctxt_->get_memory_pool().free_buffer(ptr, max_reshuffle_buffer_size_);
    }

    // if there are no communication buffers left, skip
    if (buff_sizes_.size() == 1)
        return;

    int n_buffers = buff_sizes_.size();
    // i = 0 is the initial buffer storing the matrix, so we skip this one.
    for (int i = n_buffers - 1; i >= 1; --i) {
        auto ptr = ctxt_->get_memory_pool().get_buffer_pointer(buffers_.back());
        ctxt_->get_memory_pool().free_buffer(ptr, buff_sizes_[i]);
        // remove the pointers pointing to them
        buffers_.pop_back();
    }
}

template <typename T>
Buffer<T>::~Buffer() {
    // check if communication buffers are already deallocated
    // buffers_.size() can also be 0 if the buffer was default constructed
    if (buffers_.size() > 0) {
        free_initial_buffers();
    }
}

template <typename T>
void Buffer<T>::compute_n_buckets() {
    if (strategy_->empty())
        return;
    n_buckets_ = std::vector<int>(strategy_->n_steps());
    expanded_after_ = std::vector<bool>(strategy_->n_steps());
    int prod_n_seq = 1;

    bool expanded = false;

    for (int step = strategy_->n_steps() - 1; step >= 0; --step) {
        // if the current step is sequential and this matrix was split
        // then update the product of all seq steps in
        // which this matrix was split, which represents
        // the number of buckets
        if (strategy_->sequential_step(step)) {
            if (strategy_->split(label_, step)) {
                prod_n_seq *= strategy_->divisor(step);
            }
        } else {
            // if the current matrix was expanded (i.e. NOT split)
            if (!strategy_->split(label_, step)) {
                expanded = true;
            }
        }
        n_buckets_[step] = prod_n_seq;
        expanded_after_[step] = expanded;
    }
}

template <typename T>
void Buffer<T>::init_first_split_steps() {
    int step = 0;
    first_seq_split_step = -1;
    last_first_seq_split_step = -1;
    first_par_extend_step = -1;

    while (step < strategy_->n_steps()) {
        if (strategy_->sequential_step(step) &&
            strategy_->split(label_, step)) {
            // split in seq
            if (first_par_extend_step < 0 && first_seq_split_step < 0) {
                // first_seq_step not yet found
                first_seq_split_step = step;
                last_first_seq_split_step = step;
            } else if (first_par_extend_step < 0) {
                // par step still did not occur
                last_first_seq_split_step = step;
            } else {
                break;
            }
        } else if (strategy_->parallel_step(step) &&
                   !strategy_->split(label_, step)) {
            // expanded
            if (first_par_extend_step < 0) {
                // first par step still was not found
                first_par_extend_step = step;
            } else {
                break;
            }
        }
        step++;
    }
}

template <typename T>
int Buffer<T>::buff_index_before_gemm() const {
    if (buffers_.size() == 0)
        return -1;
    if (buffers_.size() == 1)
        return 0;
    // std::cout << "par steps before gemm for " << label_ << " = " <<
    // strategy_->par_steps_before_gemm(label_) << std::endl;
    return strategy_->parallel_steps_before_gemm(label_) % 2 != 0
               ? buffers_.size() - 1
               : buffers_.size() - 2;
}

template <typename T>
T *Buffer<T>::buffer_ptr() {
    auto ptr =
        ctxt_->get_memory_pool().get_buffer_pointer(buffers_[current_buffer_]);
    return ptr;
}

template <typename T>
const T *Buffer<T>::buffer_ptr() const {
    auto ptr =
        ctxt_->get_memory_pool().get_buffer_pointer(buffers_[current_buffer_]);
    return ptr;
}

template <typename T>
const size_t Buffer<T>::buffer_size() const {
    return buff_sizes_[current_buffer_];
}

template <typename T>
int Buffer<T>::buffer_index() {
    return current_buffer_;
}

template <typename T>
void Buffer<T>::set_buffer_index(int idx) {
    current_buffer_ = idx;
}

template <typename T>
void Buffer<T>::swap_reduce_buffer_with(size_t buffer_idx) {
    std::swap(buffers_[buffer_idx], reduce_buffer_);
    std::swap(buff_sizes_[buffer_idx], max_reduce_buffer_size_);
}

template <typename T>
typename Buffer<T>::scalar_t *Buffer<T>::reshuffle_buffer_ptr() {
    if (max_reshuffle_buffer_size_ > 0)
        return ctxt_->get_memory_pool().get_buffer_pointer(reshuffle_buffer_);
    return nullptr;
}

template <typename T>
typename Buffer<T>::scalar_t *Buffer<T>::reduce_buffer_ptr() {
    if (max_reduce_buffer_size_ > 0) {
        return ctxt_->get_memory_pool().get_buffer_pointer(reduce_buffer_);
    }
    return nullptr;
}

template <typename T>
T *Buffer<T>::initial_buffer_ptr() {
    if (buffers_.size() == 0) {
        return nullptr;
    }
    return ctxt_->get_memory_pool().get_buffer_pointer(buffers_[0]);
}

template <typename T>
const T *Buffer<T>::initial_buffer_ptr() const {
    if (buffers_.size() == 0) {
        return nullptr;
    }
    return ctxt_->get_memory_pool().get_buffer_pointer(buffers_[0]);
}

template <typename T>
const size_t Buffer<T>::initial_buffer_size() const {
    if (buff_sizes_.size() == 0) {
        return 0;
    }
    return buff_sizes_[0];
}

// increases the index of the current buffer
template <typename T>
void Buffer<T>::advance_buffer() {
    // if we are at the last buffer, we then "swap" it with the pre-last buffer.
    // we do this by letting the current index point to the pre-last buffer.
    if (current_buffer_ == buffers_.size() - 1)
        current_buffer_--;
    else
        current_buffer_++;

    // should never happen
    if (current_buffer_ < 0)
        current_buffer_ = 0;
}

template <typename T>
std::vector<size_t> Buffer<T>::compute_buffer_size() {
    if (strategy_->empty()) {
        return {(size_t)mapper_->initial_size()};
    }

    Interval m(0, strategy_->m - 1);
    Interval n(0, strategy_->n - 1);
    Interval k(0, strategy_->k - 1);
    Interval P(0, strategy_->P - 1);

    // assume most memory-consuming case when beta=T{1}
    return compute_buffer_size(m, n, k, P, 0, rank_, T{1});
}

template <typename T>
std::vector<size_t> Buffer<T>::compute_buffer_size(Interval &m,
                                                   Interval &n,
                                                   Interval &k,
                                                   Interval &P,
                                                   int step,
                                                   int rank,
                                                   scalar_t beta) {
    std::vector<size_t> sizes;
    // current submatrices that are being computed
    Interval2D a_range(m, k);
    Interval2D b_range(k, n);
    Interval2D c_range(m, n);
    Interval2D range;

    // For each of P processors remember which sequential bucket we are
    // currently on
    std::vector<int> buckets = layout_->seq_buckets(P);
    // Skip all buckets that are "before" the current submatrices.
    // the relation submatrix1 <before> submatrix2 is defined in Interval2D.
    // Intuitively, this will skip all the buckets that are "above" or "on the
    // left" of the current submatrices. We say "before" because whenever we
    // split in sequential sequentially, we always first start with the "above"
    // submatrix (if the splitting is horizontal) or with the left one (if the
    // splitting is vertical). which explains the name of the relation "before".
    if (label_ == 'A') {
        range = a_range;
    } else if (label_ == 'B') {
        range = b_range;
    } else {
        range = c_range;
    }
    layout_->update_buckets(P, range);

    // check the base case
    if (n_buckets_[step] == 1) {
        compute_max_buffer_size(m, n, k, P, step, rank, beta);
        layout_->set_seq_buckets(P, buckets);
        if (expanded_after_[step]) {
            return {max_recv_buffer_size_, max_recv_buffer_size_};
        } else {
            // return {max_recv_buffer_size_, max_recv_buffer_size_};
            return {max_recv_buffer_size_};
        }
        // if (expanded_after_[step])
        //     return {max_recv_buffer_size_, max_recv_buffer_size_};
        // else
        //     return {};
    }
    // invoke a parallel or a sequential step:
    if (strategy_->sequential_step(step)) {
        int div = strategy_->divisor(step);
        int divm = strategy_->divisor_m(step);
        int divn = strategy_->divisor_n(step);
        int divk = strategy_->divisor_k(step);

        for (int i = 0; i < div; ++i) {
            Interval newm = m.subinterval(divm, divm > 1 ? i : 0);
            Interval newn = n.subinterval(divn, divn > 1 ? i : 0);
            Interval newk = k.subinterval(divk, divk > 1 ? i : 0);

            // update beta value
            scalar_t new_beta = beta;
            if (label_ == 'C' && divk > 1) {
                new_beta = 1;
                // new_beta = i == 0 && beta == 0 ? 0 : 1;
            }

            // compute substeps
            std::vector<size_t> subsizes = compute_buffer_size(
                newm, newn, newk, P, step + 1, rank, new_beta);

            // initialize the sizes vector in the first branch of sequential
            if (i == 0) {
                sizes = std::vector<size_t>(subsizes.size());
            }

            // finds the maximum buffer size for each step among all sequential
            // branches
            for (int j = 0; j < sizes.size(); ++j) {
                sizes[j] = std::max(sizes[j], subsizes[j]);
            }

            // if dividing over absent dimension, then all the branches are the
            // same so skip the rest
            if (!strategy_->split(label_, step)) {
                break;
            }
        }
        if (strategy_->split(label_, step)) {
            int max_size = 0;
            std::vector<int> block_sizes =
                layout_->sizes_inside_range(range, rank_, max_size);

            if (first_par_extend_step < 0 || step < first_par_extend_step) {
                if (step == last_first_seq_split_step) {
                    sizes.insert(sizes.begin(), max_size);
                } else {
                    sizes[0] = std::max(sizes[0], (size_t)max_size);
                }
            }
        }
    } else {
        int div = strategy_->divisor(step);
        int divm = strategy_->divisor_m(step);
        int divn = strategy_->divisor_n(step);
        int divk = strategy_->divisor_k(step);
        // processor subinterval which the current rank belongs to
        int partition_idx = P.subinterval_index(div, rank);
        Interval newP = P.subinterval(div, partition_idx);
        // intervals of M, N and K that the current rank is in charge of,
        // together with other ranks from its group.
        // (see the definition of group and offset below)
        Interval newm = m.subinterval(divm, divm > 1 ? partition_idx : 0);
        Interval newn = n.subinterval(divn, divn > 1 ? partition_idx : 0);
        Interval newk = k.subinterval(divk, divk > 1 ? partition_idx : 0);

        int offset = rank - newP.first();

        std::vector<std::vector<int>> size_before_expansion(P.length());
        std::vector<int> total_before_expansion(P.length());
        std::vector<std::vector<int>> size_after_expansion(newP.length());
        std::vector<int> total_after_expansion(newP.length());

        size_t max_size = -1;

        bool expanded = !strategy_->split(label_, step);

        if (expanded) {
            /*
             * this gives us the 2D interval of the matrix that will be
             expanded: if divm > 1 => matrix B expanded => Interval2D(k, n) if
             divn > 1 => matrix A expanded => Interval2D(m, k) if divk > 1 =>
             matrix C expanded => Interval2D(m, n)
            */
            Interval2D range;

            if (divm > 1)
                range = Interval2D(k, n);
            else if (divn > 1)
                range = Interval2D(m, k);
            else
                range = Interval2D(m, n);

            layout_->buffers_before_expansion(
                P, range, size_before_expansion, total_before_expansion);

            layout_->buffers_after_expansion(P,
                                             newP,
                                             size_before_expansion,
                                             total_before_expansion,
                                             size_after_expansion,
                                             total_after_expansion);

            // increase the buffer sizes before the substeps call
            layout_->set_sizes(newP, size_after_expansion);

            // this is the sum of sizes of all the buckets after expansion
            // that the current rank will own.
            // which is also the size of the matrix after expansion
            size_t old_size = total_before_expansion[rank - P.first()];
            size_t new_size = total_after_expansion[rank - newP.first()];
            max_size = std::max(old_size, new_size);

            int n_blocks = size_before_expansion[rank - P.first()].size();

            if (n_blocks > 1) {
                max_reshuffle_buffer_size_ =
                    std::max(max_reshuffle_buffer_size_, new_size);
            }

            // if C was expanded, then reduce was invoked
            if (label_ == 'C' && beta != scalar_t{0}) {
                int subint_index, subint_offset;
                std::tie(subint_index, subint_offset) =
                    P.locate_in_subinterval(div, rank);
                int target =
                    P.locate_in_interval(div, subint_index, subint_offset);
                max_reduce_buffer_size_ =
                    std::max(max_reduce_buffer_size_,
                             (size_t)total_before_expansion[target]);
            }
        }

        // if division by k, and we are in the branch where beta > 0, then
        // reset beta to 0, but keep in mind that on the way back from substeps
        // we will have to sum the result with the local data in C
        // this is necessary since reduction happens AFTER the substeps
        // so we cannot pass beta = 1 if the data is not present there BEFORE
        // the substeps.
        scalar_t new_beta = beta;
        if (strategy_->split_k(step) && beta != scalar_t{0}) {
            new_beta = scalar_t{0};
        }

        // invoke the substeps
        std::vector<size_t> subsizes = compute_buffer_size(
            newm, newn, newk, newP, step + 1, rank, new_beta);

        if (expanded) {
            sizes = std::vector<size_t>(subsizes.size() + 1);
            sizes[0] = max_size;
            std::copy(subsizes.begin(), subsizes.end(), sizes.begin() + 1);
            // the buffer sizes are back to the previous values
            // (the values at the beginning of this parallel step)
            layout_->set_sizes(
                newP, size_before_expansion, newP.first() - P.first());
        } else {
            sizes = subsizes;
        }
    }

    // unshift(offset);
    layout_->set_seq_buckets(P, buckets);
    return sizes;
}

template <typename T>
void Buffer<T>::compute_max_buffer_size(Interval &m,
                                        Interval &n,
                                        Interval &k,
                                        Interval &P,
                                        int step,
                                        int rank,
                                        scalar_t beta) {
    // current submatrices that are being computed
    Interval2D a_range(m, k);
    Interval2D b_range(k, n);
    Interval2D c_range(m, n);

    // For each of P processors remember which sequential bucket we are
    // currently on
    std::vector<int> buckets = layout_->seq_buckets(P);
    // Skip all buckets that are "before" the current submatrices.
    // the relation submatrix1 <before> submatrix2 is defined in Interval2D.
    // Intuitively, this will skip all the buckets that are "above" or "on the
    // left" of the current submatrices. We say "before" because whenever we
    // split sequentially, we always first start with the "above" submatrix (if
    // the splitting is horizontal) or with the left one (if the splitting is
    // vertical). which explains the name of the relation "before".
    if (label_ == 'A') {
        layout_->update_buckets(P, a_range);
    } else if (label_ == 'B') {
        layout_->update_buckets(P, b_range);
    } else {
        layout_->update_buckets(P, c_range);
    }

    // int offset = shift(buckets[rank - P.first()]);

    // invoke a parallel or a sequential step:
    if (strategy_->final_step(step)) {
        size_t max_size = 0;
        if (label_ == 'A') {
            max_size = 1LL * m.length() * k.length();
        } else if (label_ == 'B') {
            max_size = 1LL * k.length() * n.length();
        } else {
            max_size = 1LL * m.length() * n.length();
        }

        max_base_buffer_size_ = std::max(max_base_buffer_size_, max_size);

        if (max_size > max_recv_buffer_size_) {
            max_send_buffer_size_ = max_recv_buffer_size_;
            max_recv_buffer_size_ = max_size;
        } else if (max_size > max_send_buffer_size_) {
            max_send_buffer_size_ = max_size;
        }
    } else if (strategy_->sequential_step(step)) {
        int div = strategy_->divisor(step);
        int divm = strategy_->divisor_m(step);
        int divn = strategy_->divisor_n(step);
        int divk = strategy_->divisor_k(step);

        for (int i = 0; i < div; ++i) {
            Interval newm = m.subinterval(divm, divm > 1 ? i : 0);
            Interval newn = n.subinterval(divn, divn > 1 ? i : 0);
            Interval newk = k.subinterval(divk, divk > 1 ? i : 0);

            // update beta value
            scalar_t new_beta = beta;
            if (label_ == 'C' && divk > 1) {
                if (i != 0) {
                    new_beta = scalar_t{1};
                }
            }

            compute_max_buffer_size(
                newm, newn, newk, P, step + 1, rank, new_beta);

            // if dividing over absent dimension, then all the branches are the
            // same so skip the rest
            if ((label_ == 'A' && !strategy_->split_A(step)) ||
                (label_ == 'B' && !strategy_->split_B(step)) ||
                (label_ == 'C' && !strategy_->split_C(step))) {
                break;
            }
        }
    } else {
        int div = strategy_->divisor(step);
        int divm = strategy_->divisor_m(step);
        int divn = strategy_->divisor_n(step);
        int divk = strategy_->divisor_k(step);
        // processor subinterval which the current rank belongs to
        int partition_idx = P.subinterval_index(div, rank);
        Interval newP = P.subinterval(div, partition_idx);
        // intervals of M, N and K that the current rank is in charge of,
        // together with other ranks from its group.
        // (see the definition of group and offset below)
        Interval newm = m.subinterval(divm, divm > 1 ? partition_idx : 0);
        Interval newn = n.subinterval(divn, divn > 1 ? partition_idx : 0);
        Interval newk = k.subinterval(divk, divk > 1 ? partition_idx : 0);

        int offset = rank - newP.first();

        std::vector<std::vector<int>> size_before_expansion(P.length());
        std::vector<int> total_before_expansion(P.length());
        std::vector<std::vector<int>> size_after_expansion(newP.length());
        std::vector<int> total_after_expansion(newP.length());

        bool expanded = (label_ == 'A' && !strategy_->split_A(step)) ||
                        (label_ == 'B' && !strategy_->split_B(step)) ||
                        (label_ == 'C' && !strategy_->split_C(step));

        if (expanded) {
            /*
             * this gives us the 2D interval of the matrix that will be
             expanded: if divm > 1 => matrix B expanded => Interval2D(k, n) if
             divn > 1 => matrix A expanded => Interval2D(m, k) if divk > 1 =>
             matrix C expanded => Interval2D(m, n)
            */
            Interval2D range;

            if (divm > 1)
                range = Interval2D(k, n);
            else if (divn > 1)
                range = Interval2D(m, k);
            else
                range = Interval2D(m, n);

            layout_->buffers_before_expansion(
                P, range, size_before_expansion, total_before_expansion);

            layout_->buffers_after_expansion(P,
                                             newP,
                                             size_before_expansion,
                                             total_before_expansion,
                                             size_after_expansion,
                                             total_after_expansion);

            // increase the buffer sizes before the substeps
            layout_->set_sizes(newP, size_after_expansion);

            // this is the sum of sizes of all the buckets after expansion
            // that the current rank will own.
            // which is also the size of the matrix after expansion
            size_t old_size = total_before_expansion[rank - P.first()];
            size_t new_size = total_after_expansion[rank - newP.first()];
            size_t max_size = std::max(old_size, new_size);
            if (max_size > max_recv_buffer_size_) {
                max_send_buffer_size_ = max_recv_buffer_size_;
                max_recv_buffer_size_ = max_size;
            } else if (max_size > max_send_buffer_size_) {
                max_send_buffer_size_ = max_size;
            }

            int n_blocks = size_before_expansion[rank - P.first()].size();

            if (n_blocks > 1) {
                max_reshuffle_buffer_size_ =
                    std::max(max_reshuffle_buffer_size_, new_size);
            }

            // if C was expanded, then reduce was invoked
            if (label_ == 'C') {
                int subint_index, subint_offset;
                std::tie(subint_index, subint_offset) =
                    P.locate_in_subinterval(div, rank);
                int target =
                    P.locate_in_interval(div, subint_index, subint_offset);
                max_reduce_buffer_size_ =
                    std::max(max_reduce_buffer_size_,
                             (size_t)total_before_expansion[target]);
                // std::cout << "max_reduce_buffer_size = " <<
                // max_reduce_buffer_size_ << std::endl;
            }
        }

        // if division by k, and we are in the branch where beta > 0, then
        // reset beta to 0, but keep in mind that on the way back from the
        // substeps we will have to sum the result with the local data in C this
        // is necessary since reduction happens AFTER the substeps so we cannot
        // pass beta = 1 if the data is not present there BEFORE the substeps
        scalar_t new_beta = beta;
        if (strategy_->split_k(step) && beta != scalar_t{0}) {
            new_beta = 0;
        }

        // invoke the substeps
        compute_max_buffer_size(
            newm, newn, newk, newP, step + 1, rank, new_beta);

        if (expanded) {
            // the buffer sizes are back to the previous values
            // (the values at the beginning of this parallel step)
            layout_->set_sizes(
                newP, size_before_expansion, newP.first() - P.first());
        }
    }

    // unshift(offset);
    layout_->set_seq_buckets(P, buckets);
}

template <typename T>
T *Buffer<T>::operator[](const size_t index) {
    return ctxt_->get_memory_pool().get_buffer_pointer(buffers_[index]);
}

template <typename T>
T *Buffer<T>::operator[](const size_t index) const {
    return ctxt_->get_memory_pool().get_buffer_pointer(buffers_[index]);
}

template <typename T>
size_t Buffer<T>::max_send_buffer_size() const {
    return max_send_buffer_size_;
}

template <typename T>
size_t Buffer<T>::max_recv_buffer_size() const {
    return max_recv_buffer_size_;
}

// Explicit instantiations
//
template class Buffer<double>;
template class Buffer<std::complex<double>>;
template class Buffer<float>;
template class Buffer<std::complex<float>>;
template class Buffer<bfloat16>;

} // namespace cosma
