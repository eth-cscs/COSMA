#pragma once
#include <cosma/gpu/gpu_runtime_api.hpp>
#include <cosma/gpu/nccl_mapper.hpp>

#if defined(TILED_MM_CUDA)
#include <nccl.h>

#elif defined(TILED_MM_ROCM)
#include <rccl.h>

#else
#error Either TILED_MM_CUDA or TILED_MM_ROCM must be defined!
#endif


namespace cosma {
namespace gpu {
    static void check_nccl_status(ncclResult_t result) {
        if (result != ncclSuccess) {
            std::cerr << "[NCCL ERROR]: " << runtime_api::get_error_string(result) << std::endl;
            throw(std::runtime_error("NCCL ERROR"));
        }
    }

    static ncclComm_t mpi_to_nccl_comm(MPI_Comm comm) {
        if (comm == MPI_COMM_NULL) {
            return nullptr;
        }
        int my_rank, n_ranks;
        MPI_Comm_rank(comm, &my_rank);
        MPI_Comm_size(comm, &n_ranks);

        ncclUniqueId id;
        if (my_rank == 0) {
            auto status = ncclGetUniqueId(&id);
            check_nccl_status(status);
        }

        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);

        ncclComm_t nccl_comm;
        auto status = ncclCommInitRank(&nccl_comm, n_ranks, id, my_rank);
        check_nccl_status(status);

        return nccl_comm;
    }

    void free_nccl_comm(ncclComm_t nccl_comm) {
        auto status = ncclCommDestroy(nccl_comm);
        check_nccl_status(status);
    }

    // copy n*T from host to device
    template <typename T>
    void copy_to_device(const T* from, T* to, size_t n) {
        runtime_api::memcpy(to, from, n*sizeof(T), runtime_api::flag::MemcpyHostToDevice);
    }

    // copy n*T from device to host
    template <typename T>
    void copy_to_host(const T* from, T* to, size_t n) {
        runtime_api::memcpy(to, from, n*sizeof(T), runtime_api::flag::MemcpyDeviceToHost);
    }

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

    template <typename Scalar>
    void nccl_reduce(
                MPI_Comm comm,
                ncclComm_t nccl_comm,
                int rank,
                int div,
                Interval &P,
                Scalar *LC, // expanded_matrix
                Scalar *C,  // original matrix
                Scalar *reshuffle_buffer,
                Scalar *reduce_buffer,
                Scalar * d_send_buffer,
                Scalar * d_receive_buffer,
                std::vector<std::vector<int>> &c_current,
                std::vector<int> &c_total_current,
                std::vector<std::vector<int>> &c_expanded,
                std::vector<int> &c_total_expanded,
                Scalar beta,
                runtime_api::StreamType stream) {
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

        PE(multiply_communication_reduce);
        // nccl doesnt support complex numbers
        if (same_size) {
            if (!is_complex<Scalar>()) {
                auto nccl_type = nccl_mapper<Scalar>::getType();

                copy_to_device(send_pointer, d_send_pointer, nranks * recvcnts[0]);
                ncclReduceScatter(d_send_pointer,
                        d_receive_pointer,
                        recvcnts[0],
                        nccl_type,
                        ncclSum,
                        nccl_comm,
                        stream);
                copy_to_host(d_receive_pointer, receive_pointer, recvcnts[0]);

            } else {
                auto mpi_type = mpi_mapper<Scalar>::getType();
                MPI_Reduce_scatter_block(send_pointer,
                       receive_pointer,
                       recvcnts[0],
                       mpi_type,
                       MPI_SUM,
                       comm);
            }
        } else {
            auto mpi_type = mpi_mapper<Scalar>::getType();
            MPI_Reduce_scatter(send_pointer,
                    receive_pointer,
                    recvcnts.data(),
                    mpi_type,
                    MPI_SUM,
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

}  // namespace gpu
}  // namespace cosma
