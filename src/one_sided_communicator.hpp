#include "communicator.hpp"

class one_sided_communicator: public communicator {
public:
    one_sided_communicator() = default;
    one_sided_communicator(const Strategy* strategy, MPI_Comm comm): communicator(strategy, comm) {}

    void copy(Interval& P, double* in, double* out,
            std::vector<std::vector<int>>& size_before,
            std::vector<int>& total_before,
            int total_after, int step) override {
        int div = strategy_->divisor(step);
        MPI_Comm subcomm = active_comm(step);

        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);

        int local_size = total_before[relative_rank(P)];

        MPI_Info info ;
        MPI_Info_create(&info); 
        MPI_Info_set(info, "no_locks", "true");
        MPI_Info_set(info, "accumulate_ops", "same_op");
        MPI_Info_set(info, "accumulate_ordering", "none");

        MPI_Win win;
        MPI_Win_create(in, local_size*sizeof(double), sizeof(double), info, subcomm, &win);
        MPI_Info_free(&info);
        MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOPUT, win);

        int n_buckets = size_before[relative_rank(P)].size();
        double* receive_pointer = out;
        std::vector<int> bucket_offset(div);

        int displacement = 0;
        for (int bucket = 0; bucket < n_buckets; bucket++) {
            for (int rank = 0; rank < div; ++rank) {
                int target = rank_outside_ring(P, div, off, rank);
                int b_size = size_before[target][bucket];
                MPI_Get(receive_pointer + displacement, b_size, MPI_DOUBLE, rank, bucket_offset[rank], b_size, MPI_DOUBLE, win);
                bucket_offset[rank] += b_size;
                displacement += b_size;
            }
        }
        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Win_free(&win);

#ifdef DEBUG
        std::cout<<"Content of the copied matrix in rank "<<rank()<<" is now: "
            <<std::endl;
        for (int j=0; j<bucket_offset[gp]; j++) {
            std::cout<<out[j]<<" , ";
        }
        std::cout<<std::endl;
#endif
    }

    void reduce(Interval& P, double* LC, double* C,
            std::vector<std::vector<int>>& c_current,
            std::vector<int>& c_total_current,
            std::vector<std::vector<int>>& c_expanded,
            std::vector<int>& c_total_expanded,
            int beta, int step) override {
        int div = strategy_->divisor(step);
        MPI_Comm subcomm = active_comm(step);

        std::vector<int> subgroup(div);

        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);

        int n_buckets = c_expanded[off].size();
        double* send_pointer = LC;

        int target = rank_outside_ring(P, div, off, gp);
        int local_size = c_total_current[target];

        double* receive_pointer = C;
        // initilize C to 0 if beta = 0 since accumulate will do additions over this array
        if (beta == 0) {
            memset(receive_pointer, 0, local_size*sizeof(double));
        }

        MPI_Info info ;
        MPI_Info_create(&info); 
        MPI_Info_set(info, "no_locks", "true");
        MPI_Info_set(info, "accumulate_ops", "same_op");
        MPI_Info_set(info, "accumulate_ordering", "none");

        MPI_Win win;
        MPI_Win_create(receive_pointer, local_size*sizeof(double), sizeof(double),
            info, subcomm, &win);
        MPI_Info_free(&info);

        MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOSTORE, win);

        int displacement = 0;
        std::vector<int> bucket_offset(div);
        // go through the communication ring
        for (int bucket = 0; bucket < n_buckets; ++bucket) {
            for (int i = 0; i < div; ++i) {
                int target = rank_outside_ring(P, div, off, i);
                int b_size = c_current[target][bucket];

                MPI_Accumulate(send_pointer + displacement, b_size, MPI_DOUBLE, 
                    i, bucket_offset[i], b_size, MPI_DOUBLE, MPI_SUM, win);

                displacement += b_size;
                bucket_offset[i] += b_size;
            }
        }
        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Win_free(&win);
    }
};
