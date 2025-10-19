#include <cosma/bfloat16.hpp>
#include <cosma/interval.hpp>
#include <cosma/math_utils.hpp>
#include <cosma/matrix.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/profiler.hpp>
#include <cosma/strategy.hpp>
#include <cosma/two_sided_communicator.hpp>

#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <future>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <tuple>

namespace cosma {

namespace two_sided_communicator {
// two_sided_communicator() = default;
// two_sided_communicator(const Strategy* strategy, MPI_Comm comm):
//     communicator::communicator(strategy, comm) {}

/*
 * (first see the comment in communicator.hpp)
 * The idea is the following:
 *      - if only 1 block per rank should be communicated:
 *        don't allocate new space, just perform all-gather
 *
 *      - if more than 1 blocks per rank should be communicated:
 *        allocate new space and let the all-gather be performed
 *        on the level of all blocks per rank. After the communication,
 *        reshuffle the local data by putting first blocks from each rank
 * first, then all second blocks from each rank and so on.
 */
template <typename Scalar>
void copy(MPI_Comm comm,
          int rank,
          int div,
          Interval &P,
          Scalar *in,
          Scalar *out,
          Scalar *reshuffle_buffer,
          std::vector<std::vector<int>> &size_before,
          std::vector<int> &total_before,
          int total_after) {
    PE(multiply_communication_other);
    // int div = strategy_->divisor(step);
    // MPI_Comm subcomm = active_comm(step);
    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int relative_rank = rank - P.first();
    int local_size = total_before[relative_rank];

    int sum = 0;
    std::vector<int> total_size(div);
    std::vector<int> dspls(div);
    // int off = offset(P, div);

    std::vector<int> subgroup(div);
    bool same_size = true;

    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        int temp_size = total_before[target];
        dspls[i] = sum;
        sum += temp_size;
        total_size[i] = temp_size;
        same_size &= temp_size == local_size;
    }

    int n_blocks = size_before[relative_rank].size();
    Scalar *receive_pointer = n_blocks > 1 ? reshuffle_buffer : out;
    PL();

    auto mpi_type = mpi_mapper<Scalar>::getType();
    PE(multiply_communication_copy);
    if (same_size) {
        MPI_Allgather(in,
                      local_size,
                      mpi_type,
                      receive_pointer,
                      local_size,
                      mpi_type,
                      comm);
    } else {
        MPI_Allgatherv(in,
                       local_size,
                       mpi_type,
                       receive_pointer,
                       total_size.data(),
                       dspls.data(),
                       mpi_type,
                       comm);
    }
    PL();

    PE(multiply_communication_other);
    if (n_blocks > 1) {
        int index = 0;
        std::vector<int> block_offset(div);
        // order all first sequential parts of all groups first and so on..
        for (int block = 0; block < n_blocks; block++) {
            for (int rank = 0; rank < div; rank++) {
                int target = P.locate_in_interval(div, rank, off);
                int dsp = dspls[rank] + block_offset[rank];
                int b_size = size_before[target][block];
                std::copy(reshuffle_buffer + dsp,
                          reshuffle_buffer + dsp + b_size,
                          out + index);
                index += b_size;
                block_offset[rank] += b_size;
            }
        }
    }
    PL();
#ifdef DEBUG
    std::cout << "Content of the copied matrix in rank " << rank
              << " is now: " << std::endl;
    for (int j = 0; j < sum; j++) {
        std::cout << out[j] << " , ";
    }
    std::cout << std::endl;

#endif
}

template <typename Scalar>
void reduce(MPI_Comm comm,
            int rank,
            int div,
            Interval &P,
            Scalar *LC, // expanded_matrix
            Scalar *C,  // original matrix
            Scalar *reshuffle_buffer,
            Scalar *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            Scalar beta) {
    PE(multiply_communication_other);
    // int div = strategy_->divisor(step);
    // MPI_Comm subcomm = active_comm(step);

    std::vector<int> subgroup(div);

    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);
    // int gp, off;
    // std::tie(gp, off) = group_and_offset(P, div);

    // reorder the elements as:
    // first all blocks that should be sent to rank 0 then all blocks for
    // rank 1 and so on...
    int n_blocks = c_expanded[off].size();
    std::vector<int> block_offset(n_blocks);
    Scalar *send_pointer = n_blocks > 1 ? reshuffle_buffer : LC;

    int sum = 0;
    for (int i = 0; i < n_blocks; ++i) {
        block_offset[i] = sum;
        sum += c_expanded[off][i];
    }

    std::vector<int> recvcnts(div);

    bool same_size = true;
    int index = 0;
    // go through the communication ring
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        recvcnts[i] = c_total_current[target];

        same_size = same_size && recvcnts[i] == recvcnts[0];

        if (n_blocks > 1) {
            for (int block = 0; block < n_blocks; ++block) {
                int b_offset = block_offset[block];
                int b_size = c_current[target][block];
                std::copy(LC + b_offset,
                          LC + b_offset + b_size,
                          reshuffle_buffer + index);
                index += b_size;
                block_offset[block] += b_size;
            }
        }
    }

    Scalar *receive_pointer = beta != Scalar{0} ? reduce_buffer : C;
    PL();

    auto mpi_type = mpi_mapper<Scalar>::getType();
    auto mpi_sum_op = mpi_mapper<Scalar>::getSumOp();
    PE(multiply_communication_reduce);

    if (same_size) {
        MPI_Reduce_scatter_block(send_pointer,
                                 receive_pointer,
                                 recvcnts[0],
                                 mpi_type,
                                 mpi_sum_op,
                                 comm);
    } else {
        MPI_Reduce_scatter(send_pointer,
                           receive_pointer,
                           recvcnts.data(),
                           mpi_type,
                           mpi_sum_op,
                           comm);
    }
    PL();

    PE(multiply_communication_other);
    if (beta != Scalar{0}) {
        // sum up receiving_buffer with C
        for (int el = 0; el < recvcnts[gp]; ++el) {
            C[el] = beta * C[el] + reduce_buffer[el];
        }
    }
    PL();
}

template void copy<float>(MPI_Comm comm,
                          int rank,
                          int div,
                          Interval &P,
                          float *in,
                          float *out,
                          float *reshuffle_buffer,
                          std::vector<std::vector<int>> &size_before,
                          std::vector<int> &total_before,
                          int total_after);

template void copy<double>(MPI_Comm comm,
                           int rank,
                           int div,
                           Interval &P,
                           double *in,
                           double *out,
                           double *reshuffle_buffer,
                           std::vector<std::vector<int>> &size_before,
                           std::vector<int> &total_before,
                           int total_after);

template void
copy<std::complex<float>>(MPI_Comm comm,
                          int rank,
                          int div,
                          Interval &P,
                          std::complex<float> *in,
                          std::complex<float> *out,
                          std::complex<float> *reshuffle_buffer,
                          std::vector<std::vector<int>> &size_before,
                          std::vector<int> &total_before,
                          int total_after);

template void
copy<std::complex<double>>(MPI_Comm comm,
                           int rank,
                           int div,
                           Interval &P,
                           std::complex<double> *in,
                           std::complex<double> *out,
                           std::complex<double> *reshuffle_buffer,
                           std::vector<std::vector<int>> &size_before,
                           std::vector<int> &total_before,
                           int total_after);

template void copy<bfloat16>(MPI_Comm comm,
                             int rank,
                             int div,
                             Interval &P,
                             bfloat16 *in,
                             bfloat16 *out,
                             bfloat16 *reshuffle_buffer,
                             std::vector<std::vector<int>> &size_before,
                             std::vector<int> &total_before,
                             int total_after);

template void reduce<float>(MPI_Comm comm,
                            int rank,
                            int div,
                            Interval &P,
                            float *LC,
                            float *C,
                            float *reshuffle_buffer,
                            float *reduce_buffer,
                            std::vector<std::vector<int>> &c_current,
                            std::vector<int> &c_total_current,
                            std::vector<std::vector<int>> &c_expanded,
                            std::vector<int> &c_total_expanded,
                            float beta);

template void reduce<double>(MPI_Comm comm,
                             int rank,
                             int div,
                             Interval &P,
                             double *LC,
                             double *C,
                             double *reshuffle_buffer,
                             double *reduce_buffer,
                             std::vector<std::vector<int>> &c_current,
                             std::vector<int> &c_total_current,
                             std::vector<std::vector<int>> &c_expanded,
                             std::vector<int> &c_total_expanded,
                             double beta);

template void
reduce<std::complex<float>>(MPI_Comm comm,
                            int rank,
                            int div,
                            Interval &P,
                            std::complex<float> *LC,
                            std::complex<float> *C,
                            std::complex<float> *reshuffle_buffer,
                            std::complex<float> *reduce_buffer,
                            std::vector<std::vector<int>> &c_current,
                            std::vector<int> &c_total_current,
                            std::vector<std::vector<int>> &c_expanded,
                            std::vector<int> &c_total_expanded,
                            std::complex<float> beta);

template void
reduce<std::complex<double>>(MPI_Comm comm,
                             int rank,
                             int div,
                             Interval &P,
                             std::complex<double> *LC,
                             std::complex<double> *C,
                             std::complex<double> *reshuffle_buffer,
                             std::complex<double> *reduce_buffer,
                             std::vector<std::vector<int>> &c_current,
                             std::vector<int> &c_total_current,
                             std::vector<std::vector<int>> &c_expanded,
                             std::vector<int> &c_total_expanded,
                             std::complex<double> beta);

template void reduce<bfloat16>(MPI_Comm comm,
                               int rank,
                               int div,
                               Interval &P,
                               bfloat16 *LC,
                               bfloat16 *C,
                               bfloat16 *reshuffle_buffer,
                               bfloat16 *reduce_buffer,
                               std::vector<std::vector<int>> &c_current,
                               std::vector<int> &c_total_current,
                               std::vector<std::vector<int>> &c_expanded,
                               std::vector<int> &c_total_expanded,
                               bfloat16 beta);

} // end namespace two_sided_communicator

} // namespace cosma
