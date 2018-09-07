#pragma once
#include "communicator.hpp"

class two_sided_communicator: public communicator {
public:
    two_sided_communicator() = default;
    two_sided_communicator(const Strategy* strategy, MPI_Comm comm): 
        communicator::communicator(strategy, comm) {}

    /*
     * (first see the comment in communicator.hpp)
     * The idea is the following:
     *      - if only 1 block per rank should be communicated: 
     *        don't allocate new space, just perform all-gather
     *
     *      - if more than 1 blocks per rank should be communicated:
     *        allocate new space and let the all-gather be performed 
     *        on the level of all blocks per rank. After the communication,
     *        reshuffle the local data by putting first blocks from each rank first,
     *        then all second blocks from each rank and so on.
     */ 
    void copy(Interval& P, double* in, double* out, double* reshuffle_buffer,
            std::vector<std::vector<int>>& size_before,
            std::vector<int>& total_before,
            int total_after, int step) {
        int div = strategy_->divisor(step);
        MPI_Comm subcomm = active_comm(step);

        int local_size = total_before[relative_rank(P)];

        int sum = 0;
        std::vector<int> total_size(div);
        std::vector<int> dspls(div);
        int off = offset(P, div);

        std::vector<int> subgroup(div);
        bool same_size = true;

        for (int i = 0; i < div; ++i) {
            int target = rank_outside_ring(P, div, off, i);
            int temp_size = total_before[target];
            dspls[i] = sum;
            sum += temp_size;
            total_size[i] = temp_size;
            same_size &= temp_size == local_size;
        }

        int n_blocks = size_before[relative_rank(P)].size();
        double* receive_pointer = n_blocks > 1 ? reshuffle_buffer : out;

        if (same_size) {
            MPI_Allgather(in, local_size, MPI_DOUBLE, receive_pointer, local_size,
                    MPI_DOUBLE, subcomm);
        } else {
            MPI_Allgatherv(in, local_size, MPI_DOUBLE, receive_pointer,
                    total_size.data(), dspls.data(), MPI_DOUBLE, subcomm);
        }

        if (n_blocks > 1) {
            int index = 0;
            std::vector<int> block_offset(div);
            // order all first DFS parts of all groups first and so on..
            for (int block = 0; block < n_blocks; block++) {
                for (int rank = 0; rank < div; rank++) {
                    int target = rank_outside_ring(P, div, off, rank);
                    int dsp = dspls[rank] + block_offset[rank];
                    int b_size = size_before[target][block];
                    std::copy(reshuffle_buffer + dsp, reshuffle_buffer + dsp + b_size, out + index);
                    index += b_size;
                    block_offset[rank] += b_size;
                }
            }
        }
#ifdef DEBUG
        std::cout<<"Content of the copied matrix in rank "<<rank()<<" is now: "
            <<std::endl;
        for (int j=0; j<sum; j++) {
            std::cout<<out[j]<<" , ";
        }
        std::cout<<std::endl;

#endif
    }

    void reduce(Interval& P, double* LC, double* C,
            double* reshuffle_buffer, double* reduce_buffer,
            std::vector<std::vector<int>>& c_current,
            std::vector<int>& c_total_current,
            std::vector<std::vector<int>>& c_expanded,
            std::vector<int>& c_total_expanded,
            int beta, int step) {
        int div = strategy_->divisor(step);
        MPI_Comm subcomm = active_comm(step);

        std::vector<int> subgroup(div);

        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);

        // reorder the elements as:
        // first all blocks that should be sent to rank 0 then all blocks for rank 1 and so on...
        int n_blocks = c_expanded[off].size();
        std::vector<int> block_offset(n_blocks);
        double* send_pointer = n_blocks > 1 ? reshuffle_buffer : LC;

        int sum = 0;
        for (int i = 0; i < n_blocks; ++i) {
            block_offset[i] = sum;
            sum += c_expanded[off][i];
        }

        std::vector<int> recvcnts(div);

        int index = 0;
        // go through the communication ring
        for (int i = 0; i < div; ++i) {
            int target = rank_outside_ring(P, div, off, i);
            recvcnts[i] = c_total_current[target];

            if (n_blocks > 1) {
                for (int block = 0; block < n_blocks; ++block) {
                    int b_offset = block_offset[block];
                    int b_size = c_current[target][block];
                    std::copy(LC + b_offset, LC + b_offset + b_size, reshuffle_buffer + index);
                    index += b_size;
                    block_offset[block] += b_size;
                }
            }
        }

        double* receive_pointer = beta > 0 ? reduce_buffer : C;

        MPI_Reduce_scatter(send_pointer, receive_pointer, recvcnts.data(), MPI_DOUBLE, MPI_SUM, subcomm);

        if (beta > 0) {
            // sum up receiving_buffer with C
            add(C, reduce_buffer, recvcnts[gp]);
        }
    }
};
