#pragma once

#include <cosma/interval.hpp>
#include <cosma/math_utils.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>

#include <mpi.h>

#include <vector>

namespace cosma {

namespace two_sided_communicator {

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
          int total_after);

template <typename Scalar>
void reduce(MPI_Comm comm,
            void *nccl_comm_ptr,
            int rank,
            int div,
            Interval &P,
            Scalar *LC,
            Scalar *C,
            Scalar *reshuffle_buffer,
            Scalar *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            Scalar beta);

} // namespace two_sided_communicator

} // namespace cosma
