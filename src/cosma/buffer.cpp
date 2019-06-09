#include <cosma/buffer.hpp>

namespace cosma {

Buffer::Buffer(char label,
               const Strategy &strategy,
               int rank,
               Mapper *mapper,
               Layout *layout,
               bool dry_run)
    : label_(label)
    , strategy_(&strategy)
    , rank_(rank)
    , mapper_(mapper)
    , layout_(layout) {
    compute_n_buckets();
    initialize_buffers(dry_run);
}

void Buffer::initialize_buffers(bool dry_run) {
    max_base_buffer_size_ = -1;
    max_reduce_buffer_size_ = -1;
    max_reshuffle_buffer_size_ = -1;
    max_send_buffer_size_ = (long long)mapper_->initial_size();
    max_recv_buffer_size_ = (long long)mapper_->initial_size();

    init_first_split_steps();

    std::vector<long long> buff_sizes = compute_buffer_size();

    if (!dry_run) {
        // buffers_ = std::vector<std::vector<double,
        // mpi_allocator<double>>>(buff_sizes.size()+1, std::vector<double,
        // mpi_allocator<double>>());
        buffers_ = std::vector<std::vector<double, mpi_allocator<double>>>(
            buff_sizes.size(), std::vector<double, mpi_allocator<double>>());
        // std::cout << "buff sizes for " << label_ << " = " <<
        // buff_sizes.size() << std::endl;
        // buffers_[0].resize(mapper_->initial_size());
        // buffers_[0] = std::vector<double,
        // mpi_allocator<double>>(mapper_->initial_size()); ignore the first
        // buffer size since it's already allocated in the initial buffers
        // std::cout << "buffer sizes = " << std::endl;
        for (int i = 0; i < buff_sizes.size(); ++i) {
            buffers_[i].resize(buff_sizes[i]);
            // std::cout << buff_sizes[i] << ", ";
        }
        // std::cout << std::endl;
        // std::cout << "Initial size = " << mapper_->initial_size() <<
        // std::endl;
        buffers_[0].resize(
            std::max((size_t)buff_sizes[0], mapper_->initial_size()));

        // std::cout << "matrix = " << label_ << ", # buffers = " <<
        // buffers_.size() << std::endl;

        if (max_reshuffle_buffer_size_ > 0) {
            reshuffle_buffer_ = std::unique_ptr<double[]>(
                new double[max_reshuffle_buffer_size_]);
        }

        if (max_reduce_buffer_size_ > 0) {
            reduce_buffer_ =
                std::unique_ptr<double[]>(new double[max_reduce_buffer_size_]);
        }
    }

    current_buffer_ = 0;

#ifdef DEBUG
    std::cout << "Buffer sizes for matrix " << label_ << " on rank " << rank_
              << std::endl;
    std::cout << "max_reshuffle_buffer_size_ = " << max_reshuffle_buffer_size_
              << std::endl;
    std::cout << "max_reduce_buffer_size_ = " << max_reduce_buffer_size_
              << std::endl;
    std::cout << "max_send_buffer_size_ = " << max_send_buffer_size_
              << std::endl;
    std::cout << "max_recv_buffer_size_ = " << max_recv_buffer_size_
              << std::endl;
    std::cout << "max_base_buffer_size_ = " << max_base_buffer_size_
              << std::endl;
#endif

#ifdef COSMA_HAVE_GPU
    // pin the buffer that will be used in gemm
    int buff_index_to_pin = buff_index_before_gemm();
    // std::cout << "Buffer index to pin  for " << label_ << " = " <<
    // buff_index_to_pin << std::endl;
    auto &buffer_to_pin = buffers_[buff_index_to_pin];
    auto status = cudaHostRegister(buffer_to_pin.data(),
                                   buffer_to_pin.size() * sizeof(double),
                                   cudaHostRegisterDefault);
    gpu::cuda_check_status(status);
#endif
}

Buffer::~Buffer() {
#ifdef COSMA_HAVE_GPU
    // std::cout << "buffers_ size = " << buffers_.size() << std::endl;
    // unpin the buffer that will be used in gemm
    int buff_index_to_pin = buff_index_before_gemm();
    // std::cout << "Buffer index to unpin: " << buff_index_to_pin << std::endl;
    if (buff_index_to_pin >= 0) {
        auto &buffer_to_pin = buffers_[buff_index_to_pin];
        auto status = cudaHostUnregister(buffer_to_pin.data());
        gpu::cuda_check_status(status);
    }
    // std::cout << "Matrix " << label_ << ", buffers_.size() = " <<
    // buffers_.size() << ", pinned_index = " << buff_index_to_pin << std::endl;
#endif
}

void Buffer::compute_n_buckets() {
    n_buckets_ = std::vector<int>(strategy_->n_steps);
    expanded_after_ = std::vector<bool>(strategy_->n_steps);
    int prod_n_seq = 1;

    bool expanded = false;

    for (int step = strategy_->n_steps - 1; step >= 0; --step) {
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

void Buffer::init_first_split_steps() {
    int step = 0;
    first_seq_split_step = -1;
    last_first_seq_split_step = -1;
    first_par_extend_step = -1;

    while (step < strategy_->n_steps) {
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

int Buffer::buff_index_before_gemm() const {
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

std::vector<double, mpi_allocator<double>> &Buffer::buffer() {
    return buffers_[current_buffer_];
}

int Buffer::buffer_index() { return current_buffer_; }

void Buffer::set_buffer_index(int idx) { current_buffer_ = idx; }

const std::vector<double, mpi_allocator<double>> &Buffer::buffer() const {
    return buffers_[current_buffer_];
}

double *Buffer::buffer_ptr() { return buffer().data(); }

double *Buffer::reshuffle_buffer_ptr() {
    return max_reshuffle_buffer_size_ > 0 ? reshuffle_buffer_.get() : nullptr;
}

double *Buffer::reduce_buffer_ptr() {
    return max_reduce_buffer_size_ > 0 ? reduce_buffer_.get() : nullptr;
}

std::vector<double, mpi_allocator<double>> &Buffer::initial_buffer() {
    return buffers_[0];
}

const std::vector<double, mpi_allocator<double>> &
Buffer::initial_buffer() const {
    return buffers_[0];
}

double *Buffer::initial_buffer_ptr() {
    if (buffers_.size() == 0) {
        return nullptr;
    }
    return initial_buffer().data();
}

// increases the index of the current buffer
void Buffer::advance_buffer() {
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

std::vector<long long> Buffer::compute_buffer_size() {
    Interval m(0, strategy_->m - 1);
    Interval n(0, strategy_->n - 1);
    Interval k(0, strategy_->k - 1);
    Interval P(0, strategy_->P - 1);

    return compute_buffer_size(m, n, k, P, 0, rank_, strategy_->beta);
}

std::vector<long long> Buffer::compute_buffer_size(Interval &m,
                                                   Interval &n,
                                                   Interval &k,
                                                   Interval &P,
                                                   int step,
                                                   int rank,
                                                   double beta) {
    std::vector<long long> sizes;
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
            double new_beta = beta;
            if (label_ == 'C' && divk > 1) {
                new_beta = 1;
                // new_beta = i == 0 && beta == 0 ? 0 : 1;
            }

            // compute substeps
            std::vector<long long> subsizes = compute_buffer_size(
                newm, newn, newk, P, step + 1, rank, new_beta);

            // initialize the sizes vector in the first branch of sequential
            if (i == 0) {
                sizes = std::vector<long long>(subsizes.size());
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
                    sizes[0] = std::max(sizes[0], (long long)max_size);
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

        long long max_size = -1;

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
            long long old_size = total_before_expansion[rank - P.first()];
            long long new_size = total_after_expansion[rank - newP.first()];
            max_size = std::max(old_size, new_size);

            int n_blocks = size_before_expansion[rank - P.first()].size();

            if (n_blocks > 1) {
                max_reshuffle_buffer_size_ =
                    std::max(max_reshuffle_buffer_size_, new_size);
            }

            // if C was expanded, then reduce was invoked
            if (label_ == 'C' && beta > 0) {
                // if (label_ == 'C') {
                int subint_index, subint_offset;
                std::tie(subint_index, subint_offset) =
                    P.locate_in_subinterval(div, rank);
                int target =
                    P.locate_in_interval(div, subint_index, subint_offset);
                max_reduce_buffer_size_ =
                    std::max(max_reduce_buffer_size_,
                             (long long)total_before_expansion[target]);
            }
        }

        // if division by k, and we are in the branch where beta > 0, then
        // reset beta to 0, but keep in mind that on the way back from substeps
        // we will have to sum the result with the local data in C
        // this is necessary since reduction happens AFTER the substeps
        // so we cannot pass beta = 1 if the data is not present there BEFORE
        // the substeps.
        int new_beta = beta;
        if (strategy_->split_k(step) && beta > 0) {
            new_beta = 0;
        }

        // invoke the substeps
        std::vector<long long> subsizes = compute_buffer_size(
            newm, newn, newk, newP, step + 1, rank, new_beta);

        if (expanded) {
            sizes = std::vector<long long>(subsizes.size() + 1);
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

void Buffer::compute_max_buffer_size(Interval &m,
                                     Interval &n,
                                     Interval &k,
                                     Interval &P,
                                     int step,
                                     int rank,
                                     double beta) {
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
        long long max_size = 0;
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
            double new_beta = beta;
            if (label_ == 'C' && divk > 1) {
                new_beta = i == 0 && beta == 0 ? 0 : 1;
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
            long long old_size = total_before_expansion[rank - P.first()];
            long long new_size = total_after_expansion[rank - newP.first()];
            long long max_size = std::max(old_size, new_size);
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
                             (long long)total_before_expansion[target]);
                // std::cout << "max_reduce_buffer_size = " <<
                // max_reduce_buffer_size_ << std::endl;
            }
        }

        // if division by k, and we are in the branch where beta > 0, then
        // reset beta to 0, but keep in mind that on the way back from the
        // substeps we will have to sum the result with the local data in C this
        // is necessary since reduction happens AFTER the substeps so we cannot
        // pass beta = 1 if the data is not present there BEFORE the substeps
        int new_beta = beta;
        if (strategy_->split_k(step) && beta > 0) {
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

std::vector<double, mpi_allocator<double>> &Buffer::
operator[](const std::vector<double, mpi_allocator<double>>::size_type index) {
    return buffers_[index];
}

std::vector<double, mpi_allocator<double>> Buffer::operator[](
    const std::vector<double, mpi_allocator<double>>::size_type index) const {
    return buffers_[index];
}

long long Buffer::max_send_buffer_size() const { return max_send_buffer_size_; }
long long Buffer::max_recv_buffer_size() const { return max_recv_buffer_size_; }
} // namespace cosma
