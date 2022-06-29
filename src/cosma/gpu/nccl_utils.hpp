#pragma once
#include <vector>

#include <cosma/interval.hpp>
#include <cosma/context.hpp>

#include <mpi.h>

#if defined(TILED_MM_CUDA)
#include <nccl.h>

#elif defined(TILED_MM_ROCM)
#include <rccl.h>

#else
#error Either TILED_MM_CUDA or TILED_MM_ROCM must be defined!
#endif

namespace cosma {
namespace gpu {
    void check_nccl_status(ncclResult_t result);

    ncclComm_t mpi_to_nccl_comm(MPI_Comm comm);

    void free_nccl_comm(ncclComm_t nccl_comm);

    template <typename Scalar>
    void nccl_copy(
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
    void nccl_reduce(
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
