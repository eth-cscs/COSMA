#include "communicator.hpp"

namespace communicator {
    void initialize(int * argc, char ***argv) {
        MPI_Init(argc, argv);
    }

    int rank() {
        int r;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }

    int size() {
        int s;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
    }

    void free(MPI_Comm comm) {
        MPI_Comm_free(&comm);
    }

    void finalize() {
        MPI_Finalize();
    }

    void barrier(MPI_Comm comm) {
        MPI_Barrier(comm);
    }

    int offset(Interval& P, int div, int r) {
        int subset_size = P.length() / div;
        r = relative_rank(P, r);
        int gp = r / subset_size;
        int off = r - gp * subset_size;
        return off;
    }

    int group(Interval& P, int div, int r) {
        int subset_size = P.length() / div;
        int gp = relative_rank(P, r) / subset_size;
        return gp;
    }

    std::pair<int, int> group_and_offset(Interval& P, int div, int r) {
        int subset_size = P.length() / div;
        r = relative_rank(P, r);
        int gp = r / subset_size;
        int off = r - gp * subset_size;
        return {gp, off};
    }

    MPI_Comm split_in_groups(MPI_Comm comm, Interval& P, int div, int r) {
        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div, r);

        MPI_Comm new_comm;
        MPI_Comm_split(comm, gp, off, &new_comm);
        std::cout << "rank = " << r << " gp = " << gp << " offset = " << off << std::endl;
        return new_comm;
    }

    MPI_Comm split_in_comm_rings(MPI_Comm comm, Interval& P, int div, int r) {
        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div, r);

        MPI_Comm new_comm;
        MPI_Comm_split(comm, off, gp, &new_comm);
        std::cout << "rank = " << r << " gp = " << gp << " offset = " << off << std::endl;
        return new_comm;
    }

    int rank_inside_ring(Interval& P, int div, int global_rank) {
        return group(P, div, global_rank);
    }

    int rank_outside_ring(Interval& P, int div, int off, int i) {
        int subset_size = P.length() / div;
        return i * subset_size + off;
    }

    int relative_rank(Interval& P, int r) {
        return r - P.first();
    }

    void copy(int div, Interval& P, double* in, double* out,
              std::vector<std::vector<int>>& size_before,
              std::vector<int>& total_before,
              int total_after, MPI_Comm comm) {
        int local_size = total_before[relative_rank(P)];

        int sum = 0;
        std::vector<int> total_size(div);
        std::vector<int> dspls(div);
        int off = offset(P, div);
        std::cout << "P divided by " << div << " of rank " << rank() << " gives offset = " << off << std::endl;

        for (int i = 0; i < div; ++i) {
            int target = rank_outside_ring(P, div, off, i);
            int temp_size = total_before[target];
            dspls[i] = sum;
            sum += temp_size;
            total_size[i] = temp_size;
        }

        std::vector<double> receiving_buffer(total_after);

        MPI_Comm subcomm = split_in_comm_rings(comm, P, div);

        MPI_Allgatherv(in, local_size, MPI_DOUBLE, receiving_buffer.data(),
                       total_size.data(), dspls.data(), MPI_DOUBLE, subcomm);
        std::cout << "rank = " << rank() << " Relative rank = " << relative_rank(P) << std::endl;
        int n_buckets = size_before[relative_rank(P)].size();
        int index = 0;
        std::vector<int> bucket_offset(div);
        // order all first DFS parts of all groups first and so on..
        for (int bucket = 0; bucket < n_buckets; bucket++) {
            for (int rank = 0; rank < div; rank++) {
                int target = rank_outside_ring(P, div, off, rank);
                int dsp = dspls[rank] + bucket_offset[rank];
                int b_size = size_before[target][bucket];
                std::copy(receiving_buffer.begin() + dsp, receiving_buffer.begin() + dsp + b_size, out + index);
                index += b_size;
                bucket_offset[rank] += b_size;
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
        communicator::free(subcomm);
    }

    void reduce(int div, Interval& P, double* LC, double* C,
                std::vector<std::vector<int>>& c_current,
                std::vector<int>& c_total_current,
                std::vector<std::vector<int>>& c_expanded,
                std::vector<int>& c_total_expanded,
                int beta,
                MPI_Comm comm) {
        MPI_Comm subcomm = split_in_comm_rings(comm, P, div);
        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);

        // reorder the elements as:
        // first all buckets that should be sent to rank 0 then all buckets for rank 1 and so on...
        std::vector<double> send_buffer(c_total_expanded[off]);
        int n_buckets = c_expanded[off].size();
        std::vector<int> bucket_offset(n_buckets);

        int sum = 0;
        for (int i = 0; i < n_buckets; ++i) {
            bucket_offset[i] = sum;
            sum += c_expanded[off][i];
        }

        std::vector<int> recvcnts(div);

        int index = 0;
        // go through the communication ring
        for (int i = 0; i < div; i++) {
            int target = rank_outside_ring(P, div, off, i);
            recvcnts[i] = c_total_current[target];

            for (int bucket = 0; bucket < n_buckets; ++bucket) {
                int b_offset = bucket_offset[bucket];
                int b_size = c_current[target][bucket];
                std::copy(LC + b_offset, LC + b_offset + b_size, send_buffer.begin() + index);
                index += b_size;
                bucket_offset[bucket] += b_size;
            }
        }

        double* receiving_buffer;
        if (beta == 0) {
            receiving_buffer = C;
        } else {
            receiving_buffer = (double*) malloc(sizeof(double) * recvcnts[gp]);
        }

        MPI_Reduce_scatter(send_buffer.data(), receiving_buffer, recvcnts.data(), MPI_DOUBLE, MPI_SUM, subcomm);

        if (beta > 0) {
            for (int i = 0; i < recvcnts[gp]; ++i) {
                C[i] += receiving_buffer[i];
            }
        }

        communicator::free(subcomm);
    }

};

