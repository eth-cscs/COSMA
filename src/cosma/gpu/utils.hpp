#pragma once
#include <complex>
#include <vector>
#include <iostream>

#include <cosma/interval.hpp>
#include <cosma/context.hpp>
#include <cosma/gpu/gpu_runtime_api.hpp>
#include <cosma/gpu/nccl_mapper.hpp>
#include <cosma/profiler.hpp>
#include <cosma/mpi_mapper.hpp>

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

    void check_runtime_status(runtime_api::StatusType status);

    // copy n*T from host to device
    // If a cuda stream is passed as the final argument the copy will be performed
    // asynchronously in the specified stream, otherwise it will be serialized in
    // the default (NULL) stream
    template <typename T>
    void copy_to_device_async(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        //cudaDeviceSynchronize();
        // auto status = cudaGetLastError();
        // if(status != cudaSuccess) {
        //    std::cout << "error: CUDA kernel launch:"
        //    << cudaGetErrorString(status) << std::endl;
        //    throw(std::runtime_error("CUDA ERROR"));
        //}

        auto status = runtime_api::memcpy_async(to, from, n * sizeof(T),
                                                runtime_api::flag::MemcpyHostToDevice, stream);
        check_runtime_status(status);
    }

    // copy n*T from device to host
    // If a cuda stream is passed as the final argument the copy will be performed
    // asynchronously in the specified stream, otherwise it will be serialized in
    // the default (NULL) stream
    template <typename T>
    void copy_to_host_async(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        auto status = runtime_api::memcpy_async(to, from, n * sizeof(T),
                                                runtime_api::flag::MemcpyDeviceToHost, stream);
        check_runtime_status(status);
    }

    template <typename T>
    void copy_to_host(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        auto status = runtime_api::memcpy(to, from, n * sizeof(T),
                                          runtime_api::flag::MemcpyDeviceToHost, stream);
        check_runtime_status(status);
    }

    template <typename T>
    void copy_device_to_device_async(const T* from, T* to, size_t n, runtime_api::StreamType stream=NULL) {
        auto status = runtime_api::memcpy_async(to, from, n * sizeof(T),
                                                runtime_api::flag::MemcpyDeviceToDevice, stream);
        check_runtime_status(status);
    }

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
                size_t step) {
        PE(multiply_communication_other);
        auto mpi_comm = ctx->get_cosma_comm()->active_comm(step);
        auto nccl_comm = ctx->get_cosma_comm()->active_nccl_comm(step);

        int rank = ctx->get_cosma_comm()->rank();
        int div = ctx->get_cosma_comm()->get_strategy()->divisor(step);

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
        int sum = 0;
        for (int i = 0; i < n_blocks; ++i) {
            block_offset[i] = sum;
            sum += c_expanded[off][i];
        }

        std::vector<int> recvcnts(div);
        int max_block_size = 0;
        for (int i = 0; i < div; ++i) {
            int target = P.locate_in_interval(div, i, off);
            recvcnts[i] = c_total_current[target];
            // the max block size (used to determine the padding)
            max_block_size = std::max(max_block_size, recvcnts[i]);
        }

        // here is the result of matrix multiplication on GPU
        Scalar* d_LC = ctx->get_gpu_context()->get_full_device_buffer_c().data();

        // this will only resize the buffer if not already allocated
        ctx->get_memory_pool().allocate_device_send_buffer(div * max_block_size);
        Scalar* d_reshuffle_buffer = ctx->get_memory_pool().device_send_buffer.data();

        ctx->get_memory_pool().allocate_device_receive_buffer(max_block_size);
        Scalar* d_receive_pointer = ctx->get_memory_pool().device_receive_buffer.data();

        auto stream = ctx->nccl_stream.stream();

        int index = 0;
        // go through the communication ring
        for (int i = 0; i < div; ++i) {
            int target = P.locate_in_interval(div, i, off);

            for (int block = 0; block < n_blocks; ++block) {
                int b_offset = block_offset[block];
                int b_size = c_current[target][block];
                // reshuffle directly into the gpu buffer
                gpu::copy_device_to_device_async(d_LC + b_offset, 
                                                 d_reshuffle_buffer + index, 
                                                 b_size, stream);
                // pad with 0s if not all the blocks are the same
                if (b_size < max_block_size) {
                    gpu::runtime_api::memset_async(d_reshuffle_buffer + index + b_size, 0, max_block_size - b_size);
                }
                index += max_block_size;
                block_offset[block] += b_size;
            }
        }

        Scalar *receive_pointer = beta != Scalar{0} ? reduce_buffer : C;
        PL();

        PE(multiply_communication_reduce);
        auto nccl_type = nccl_mapper<Scalar>::getType();
        ncclReduceScatter(d_reshuffle_buffer,
                d_receive_pointer,
                max_block_size,
                nccl_type,
                ncclSum,
                nccl_comm,
                stream);
        gpu::copy_to_host_async(d_receive_pointer, receive_pointer, recvcnts[gp], stream);

        // wait for the result on the host
        gpu::runtime_api::stream_synchronize(stream);

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

}  // namespace gpu
}  // namespace cosma
