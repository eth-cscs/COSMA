#include <costa/grid2grid/communication_data.hpp>

#include <complex>
#include <omp.h>

namespace costa {
// *********************
//     MESSAGE
// *********************
template <typename T>
message<T>::message(block<T> b, int rank)
    : b(b)
    , rank(rank) {}

template <typename T>
block<T> message<T>::get_block() const {
    return b;
}

template <typename T>
int message<T>::get_rank() const {
    return rank;
}

// implementing comparator
template <typename T>
bool message<T>::operator<(const message<T> &other) const {
    return get_rank() < other.get_rank() ||
           (get_rank() == other.get_rank() && b < other.get_block()); 
}

template <typename T>
void communication_data<T>::partition_messages() {
    if (mpi_messages.size() == 0) 
        return;

    int pivot = -1; 
    for (int i = 0; i < mpi_messages.size(); ++i) {
        int rank = mpi_messages[i].get_rank();
        if (pivot != rank) {
            pivot = rank;
            package_ticks.push_back(i);
        }
    }
    package_ticks.push_back(mpi_messages.size());
}

// ************************
//   COMMUNICATION DATA
// ************************
template <typename T>
communication_data<T>::communication_data(std::vector<message<T>> &messages,
                                          int rank, int n_ranks)
    : n_ranks(n_ranks)
    , my_rank(rank) {
    // std::cout << "constructor of communciation data invoked" << std::endl;
    dspls = std::vector<int>(n_ranks);
    counts = std::vector<int>(n_ranks);
    mpi_messages.reserve(messages.size());
    offset_per_message.reserve(messages.size());

    int offset = 0;

    int prev_rank = -1;

    for (unsigned i = 0; i < messages.size(); ++i) {
        const auto &m = messages[i];
        int target_rank = m.get_rank();
        block<T> b = m.get_block();
        assert(b.non_empty());

        // if the message should be communicated to 
        // a different rank
        if (target_rank != my_rank) {
            mpi_messages.push_back(m);
            offset_per_message.push_back(offset);
            offset += b.total_size();
            counts[target_rank] += b.total_size();
            total_size += b.total_size();
            prev_rank = target_rank;
        } else {
            local_messages.push_back(m);
        }
    }

    buffer = std::unique_ptr<T[]>(new T[total_size]);
    for (unsigned i = 1; i < (unsigned)n_ranks; ++i) {
        dspls[i] = dspls[i - 1] + counts[i - 1];
    }

    n_packed_messages = 0;
    for (unsigned i = 0; i < (unsigned) n_ranks; ++i) {
        if (counts[i]) {
            ++n_packed_messages;
        }
    }

    partition_messages();
}

template <typename T>
void copy_block_to_buffer(block<T> b, T *dest_ptr) {
    // std::cout << "copy block->buffer: " << b << std::endl;
    // std::cout << "copy block->buffer" << std::endl;
    if (!b.transpose_on_copy)
        memory::copy2D(b.size(), b.data, b.stride, dest_ptr, b.n_rows());
    else {
        // stride in the destination
        // is the number of columns
        // because block b will be transposed
        // in the buffer without any stride
        // (we make the buffer packed)
        int dest_stride = b.n_rows();
        memory::copy_and_transpose(b, dest_ptr, dest_stride);
        // b.stride = b.n_cols();
    }
}

template <typename T>
void copy_block_from_buffer_and_scale(T *src_ptr, block<T> &b, T alpha, T beta) {
    // std::cout << "copy buffer->block" << std::endl;
    memory::copy2D_and_scale(b.size(), src_ptr, b.n_rows(), b.data, b.stride, 
                             alpha, beta);
}

template <typename T>
void copy_block_from_buffer(T *src_ptr, block<T> &b) {
    // std::cout << "copy buffer->block" << std::endl;
    memory::copy2D(b.size(), src_ptr, b.n_rows(), b.data, b.stride);
}

template <typename T>
void communication_data<T>::copy_to_buffer() {
    if (mpi_messages.size()) {
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < mpi_messages.size(); ++i) {
            const auto &m = mpi_messages[i];
            block<T> b = m.get_block();
            copy_block_to_buffer(b, data() + offset_per_message[i]);
        }
    }
}

template <typename T>
void communication_data<T>::copy_to_buffer(int idx) {
    assert(idx >= 0 && idx+1 < package_ticks.size());
    if (package_ticks[idx+1] - package_ticks[idx]) {
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = package_ticks[idx]; i < package_ticks[idx+1]; ++i) {
            const auto &m = mpi_messages[i];
            block<T> b = m.get_block();
            copy_block_to_buffer(b, data() + offset_per_message[i]);
        }
    }
}

template <typename T>
void communication_data<T>::copy_from_buffer(int idx) {
    assert(idx >= 0 && idx+1 < package_ticks.size());
    if (package_ticks[idx+1] - package_ticks[idx]) {
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = package_ticks[idx]; i < package_ticks[idx+1]; ++i) {
            const auto &m = mpi_messages[i];
            block<T> b = m.get_block();
            if (m.alpha != T{1} || m.beta != T{0}) {
                copy_block_from_buffer_and_scale(data() + offset_per_message[i], b, 
                                                 m.alpha, m.beta);
            } else {
                copy_block_from_buffer(data() + offset_per_message[i], b);
            }
        }
    }
}

template <typename T>
void communication_data<T>::copy_from_buffer() {
    if (mpi_messages.size()) {
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < mpi_messages.size(); ++i) {
            const auto &m = mpi_messages[i];
            block<T> b = m.get_block();
            if (m.alpha != T{1} || m.beta != T{0}) {
                copy_block_from_buffer_and_scale(data() + offset_per_message[i], b,
                                                 m.alpha, m.beta);
            } else {
                copy_block_from_buffer(data() + offset_per_message[i], b);
            }
        }
    }
}

template <typename T>
T *communication_data<T>::data() {
    return buffer.get();
}

template <typename T>
void copy_block_to_block_and_scale(block<T>& src, block<T>& dest, 
                                   T alpha, T beta) {
    // std::cout << "copy buffer->block" << std::endl;
    if (!src.transpose_on_copy) {
        memory::copy2D_and_scale(src.size(), src.data, src.stride, 
                                 dest.data, dest.stride,
                                 alpha, beta);
    } else {
        // transpose and conjugate if necessary while copying
        memory::copy_transpose_and_scale(src, dest.data, dest.stride,
                                             alpha, beta);
    }
}

template <typename T>
void copy_block_to_block(block<T>& src, block<T>& dest) {
    // std::cout << "copy buffer->block" << std::endl;
    if (!src.transpose_on_copy) {
        memory::copy2D(src.size(), src.data, src.stride, 
                       dest.data, dest.stride);
    } else {
        // transpose and conjugate if necessary while copying
        memory::copy_and_transpose(src, dest.data, dest.stride);
    }
}

template <typename T>
void copy_local_blocks(std::vector<message<T>>& from,
                       std::vector<message<T>>& to) {
    assert(from.size() == to.size());
    if (from.size() > 0) {
#pragma omp parallel for
        for (unsigned i = 0u; i < from.size(); ++i) {
            assert(from[i].alpha == to[i].alpha);
            assert(from[i].beta == to[i].beta);
            auto block_src = from[i].get_block();
            auto block_dest = to[i].get_block();
            assert(block_src.non_empty());
            assert(block_dest.non_empty());
            assert(block_src.total_size() == block_dest.total_size());
            // destination block cannot be transposed
            assert(!block_dest.transpose_on_copy);

            const auto& alpha = from[i].alpha;
            const auto& beta = from[i].beta;

            if (alpha != T{1} || beta != T{0}) {
                copy_block_to_block_and_scale(block_src, block_dest,
                                    alpha, beta);
            } else {
                copy_block_to_block(block_src, block_dest);
            }
    }
    }
}

// template instantiation for communication_data
template class communication_data<double>;
template class communication_data<std::complex<double>>;
template class communication_data<float>;
template class communication_data<std::complex<float>>;

// template instantiation for message
template class message<double>;
template class message<std::complex<double>>;
template class message<float>;
template class message<std::complex<float>>;

// template instantiation for copy_local_blocks
template void
copy_local_blocks(std::vector<message<double>>& from, 
                  std::vector<message<double>>& to);
template void
copy_local_blocks(std::vector<message<float>>& from, 
                  std::vector<message<float>>& to);
template void
copy_local_blocks(std::vector<message<std::complex<float>>>& from, 
                  std::vector<message<std::complex<float>>>& to);
template void
copy_local_blocks(std::vector<message<std::complex<double>>>& from, 
                  std::vector<message<std::complex<double>>>& to);
} // namespace costa
