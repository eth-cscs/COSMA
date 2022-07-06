#include <complex>
#include <iostream>

#include <cosma/communicator.hpp>
#include <cosma/gpu/utils.hpp>
#include <cosma/gpu/gpu_aware_mpi_utils.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/profiler.hpp>

template <typename Scalar>
void cosma::gpu::gpu_aware_mpi_copy(
            cosma_context<Scalar> *ctx,
            Interval &P,
            Scalar * in, // original_matrix
            Scalar * out,  // expanded matrix
            Scalar *reshuffle_buffer,
            std::vector<std::vector<int>>& size_before,
            std::vector<int> &total_before,
            int total_after,
            size_t step) {
    PE(multiply_communication_other);
    auto mpi_comm = ctx->get_cosma_comm()->active_comm(step);

    int rank = ctx->get_cosma_comm()->rank();
    int div = ctx->get_cosma_comm()->get_strategy().divisor(step);

    int gp, off;
    std::tie(gp, off) = P.locate_in_subinterval(div, rank);

    int relative_rank = rank - P.first();
    int local_size = total_before[relative_rank];

    int sum = 0;
    std::vector<int> total_size(div);
    std::vector<int> dspls(div);

    std::vector<int> subgroup(div);
    bool same_size = true;

    int max_block_size = 0;
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        int temp_size = total_before[target];
        dspls[i] = sum;
        sum += temp_size;
        total_size[i] = temp_size;
        same_size &= temp_size == local_size;
        max_block_size = std::max(max_block_size, temp_size);
    }

    int n_blocks = size_before[relative_rank].size();

    // this will only resize the buffer if not already allocated
    ctx->get_memory_pool().allocate_device_receive_buffer(max_block_size);
    Scalar* d_send_pointer = ctx->get_memory_pool().device_receive_buffer.data();

    ctx->get_memory_pool().allocate_device_send_buffer(div * max_block_size);
    Scalar* d_receive_pointer = ctx->get_memory_pool().device_send_buffer.data();

    auto stream = ctx->gpu_stream.stream();

    // copy input matrix to device
    gpu::copy_to_device_async(in, d_send_pointer, local_size, stream);

    PL();

    PE(multiply_communication_reduce);
    auto mpi_type = mpi_mapper<Scalar>::getType();

    // since it's not possible to pass the stream to MPI
    // to perform the collective on that stream
    // we have to make sure the data is copied to gpu
    // before MPI function is called
    gpu::runtime_api::stream_synchronize(stream);

    MPI_Allgather(d_send_pointer,
            max_block_size,
            mpi_type,
            d_receive_pointer,
            max_block_size,
            mpi_type,
            mpi_comm);

    // make sure MPI has finished
    // gpu::runtime_api::device_synchronize();

    PE(multiply_communication_other);
    int index = 0;
    std::vector<int> block_offset(div);
    // order all first sequential parts of all groups first and so on..
    for (int block = 0; block < n_blocks; block++) {
        for (int rank = 0; rank < div; rank++) {
            int target = P.locate_in_interval(div, rank, off);
            int dsp = dspls[rank] + block_offset[rank];
            int b_size = size_before[target][block];
            gpu::copy_to_host_async(
                d_receive_pointer + rank * max_block_size + block_offset[rank],
                out + index, 
                b_size,
                stream);
            index += b_size;
            block_offset[rank] += b_size;
        }
    }
    PL();

    // wait for the result on the host
    gpu::runtime_api::stream_synchronize(stream);

    PL();
}

template <typename Scalar>
void cosma::gpu::gpu_aware_mpi_reduce(
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
            bool copy_c_back) {
    PE(multiply_communication_other);
    auto mpi_comm = ctx->get_cosma_comm()->active_comm(step);

    int rank = ctx->get_cosma_comm()->rank();
    int div = ctx->get_cosma_comm()->get_strategy().divisor(step);

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
    int min_block_size = recvcnts[0];
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);
        recvcnts[i] = c_total_current[target];
        // the max block size (used to determine the padding)
        max_block_size = std::max(max_block_size, recvcnts[i]);
        min_block_size = std::min(min_block_size, recvcnts[i]);
    }

    bool same_blocks = max_block_size == min_block_size;

    // here is the result of matrix multiplication on GPU
    Scalar* d_LC = LC;
    if (!copy_c_back) {
        d_LC = ctx->get_gpu_context()->get_full_device_buffer_c().data();
    }

    // this will only resize the buffer if not already allocated
    ctx->get_memory_pool().allocate_device_send_buffer(div * max_block_size);
    Scalar* d_reshuffle_buffer = ctx->get_memory_pool().device_send_buffer.data();

    ctx->get_memory_pool().allocate_device_receive_buffer(max_block_size);
    Scalar* d_receive_pointer = ctx->get_memory_pool().device_receive_buffer.data();

    auto stream = ctx->gpu_stream.stream();

    // set all to 0s, so that we don't have to pad each block with 0s up to max_block_size
    /*
    if (!same_blocks) {
        gpu::runtime_api::memset_async(d_reshuffle_buffer, 0, div * max_block_size, stream);
    }
    */

    std::vector<int> blocks_offset_per_group(div, 0);
    // go through the communication ring
    for (int i = 0; i < div; ++i) {
        int target = P.locate_in_interval(div, i, off);

        for (int block = 0; block < n_blocks; ++block) {
            int b_offset = block_offset[block];
            int b_size = c_current[target][block];
            // reshuffle directly into the gpu buffer
            if (!copy_c_back) {
                gpu::copy_device_to_device_async(d_LC + b_offset, 
                                                 d_reshuffle_buffer + i * max_block_size + blocks_offset_per_group[i],
                                                 b_size, stream);
            } else {
                gpu::copy_to_device_async(d_LC + b_offset, 
                                          d_reshuffle_buffer + i * max_block_size + blocks_offset_per_group[i],
                                          b_size, stream);
            }
            block_offset[block] += b_size;
            blocks_offset_per_group[i] += b_size;
        }
    }

    Scalar *receive_pointer = beta != Scalar{0} ? reduce_buffer : C;
    PL();

    PE(multiply_communication_reduce);
    auto mpi_type = mpi_mapper<Scalar>::getType();

    // since it's not possible to pass the stream to MPI
    // to perform the collective on that stream
    // we have to make sure the data is copied to gpu
    // before MPI function is called
    gpu::runtime_api::stream_synchronize(stream);

    MPI_Reduce_scatter_block(d_reshuffle_buffer,
            d_receive_pointer,
            max_block_size,
            mpi_type,
            MPI_SUM,
            mpi_comm);

    // make sure MPI has finished
    // gpu::runtime_api::device_synchronize();

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

// template instantiation for gpu_aware_mpi_reduce
template void cosma::gpu::gpu_aware_mpi_reduce<float>(
            cosma_context<float> *ctx,
            Interval &P,
            float *LC, // expanded_matrix
            float *C,  // original matrix
            float *reshuffle_buffer,
            float *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            float beta,
            size_t step,
            bool copy_c_back);

template void cosma::gpu::gpu_aware_mpi_reduce<double>(
            cosma_context<double> *ctx,
            Interval &P,
            double *LC, // expanded_matrix
            double *C,  // original matrix
            double *reshuffle_buffer,
            double *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            double beta,
            size_t step,
            bool copy_c_back);

template void cosma::gpu::gpu_aware_mpi_reduce<std::complex<float>>(
            cosma_context<std::complex<float>> *ctx,
            Interval &P,
            std::complex<float> *LC, // expanded_matrix
            std::complex<float> *C,  // original matrix
            std::complex<float> *reshuffle_buffer,
            std::complex<float> *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            std::complex<float> beta,
            size_t step,
            bool copy_c_back);

template void cosma::gpu::gpu_aware_mpi_reduce<std::complex<double>>(
            cosma_context<std::complex<double>> *ctx,
            Interval &P,
            std::complex<double> *LC, // expanded_matrix
            std::complex<double> *C,  // original matrix
            std::complex<double> *reshuffle_buffer,
            std::complex<double> *reduce_buffer,
            std::vector<std::vector<int>> &c_current,
            std::vector<int> &c_total_current,
            std::vector<std::vector<int>> &c_expanded,
            std::vector<int> &c_total_expanded,
            std::complex<double> beta,
            size_t step,
            bool copy_c_back);

// template instantiation for gpu_aware_mpi_copy
template void cosma::gpu::gpu_aware_mpi_copy<float>(
            cosma_context<float> *ctx,
            Interval &P,
            float * in, // original_matrix
            float * out,  // expanded matrix
            float *reshuffle_buffer,
            std::vector<std::vector<int>>& size_before,
            std::vector<int> &total_before,
            int total_after,
            size_t step);

template void cosma::gpu::gpu_aware_mpi_copy<double>(
            cosma_context<double> *ctx,
            Interval &P,
            double * in, // original_matrix
            double * out,  // expanded matrix
            double *reshuffle_buffer,
            std::vector<std::vector<int>>& size_before,
            std::vector<int> &total_before,
            int total_after,
            size_t step);

template void cosma::gpu::gpu_aware_mpi_copy<std::complex<float>>(
            cosma_context<std::complex<float>> *ctx,
            Interval &P,
            std::complex<float> * in, // original_matrix
            std::complex<float> * out,  // expanded matrix
            std::complex<float> *reshuffle_buffer,
            std::vector<std::vector<int>>& size_before,
            std::vector<int> &total_before,
            int total_after,
            size_t step);

template void cosma::gpu::gpu_aware_mpi_copy<std::complex<double>>(
            cosma_context<std::complex<double>> *ctx,
            Interval &P,
            std::complex<double> * in, // original_matrix
            std::complex<double> * out,  // expanded matrix
            std::complex<double> *reshuffle_buffer,
            std::vector<std::vector<int>>& size_before,
            std::vector<int> &total_before,
            int total_after,
            size_t step);

