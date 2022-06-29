#pragma once
#include <vector>

#include <cosma/interval.hpp>
#include <cosma/context.hpp>

#include <mpi.h>

namespace cosma {
namespace gpu {
    template <typename Scalar>
    void gpu_aware_mpi_copy(
                cosma_context<Scalar> *ctx,
                Interval &P,
                Scalar * in, // original_matrix
                Scalar * out,  // expanded matrix
                Scalar *reshuffle_buffer,
                std::vector<std::vector<int>>& size_before,
                std::vector<int> &total_before,
                int total_after,
                size_t step);

    template <typename Scalar>
    void gpu_aware_mpi_reduce(
                cosma_context<Scalar> *ctx,
                Interval &P,
                Scalar *LC, // expanded_matrix
                Scalar *C,  // original matrix
                Scalar *reshuffle_buffer,
                Scalar *reduce_buffer,
                std::vector<std::vector<int>> &c_current,
                std::vector<int> &c_total_current,
                std::vector<std::vector<int>> &c_expanded,
                std::vector<int> &c_total_expanded,
                Scalar beta,
                size_t step,
                bool copy_c_back);

}  // namespace gpu
}  // namespace cosma
