#include "communicator.hpp"

class one_sided_communicator: public communicator {
public:
    one_sided_communicator() = default;
    one_sided_communicator(const Strategy* strategy, MPI_Comm comm): communicator(strategy, comm) {}

    MPI_Win create_window(double* pointer, int size, int step) {
        MPI_Comm comm = active_comm(step);

        MPI_Info info ;
        MPI_Info_create(&info); 
        MPI_Info_set(info, "no_locks", "true");
        MPI_Info_set(info, "accumulate_ops", "same_op");
        MPI_Info_set(info, "accumulate_ordering", "none");

        MPI_Win win;
        MPI_Win_create(pointer, size*sizeof(double), sizeof(double), info, comm, &win);

        MPI_Info_free(&info);

        return win;
    }

    void copy(Interval& P, double* in, double* out,
            std::vector<std::vector<int>>& size_before,
            std::vector<int>& total_before,
            int total_after, int step) override {
        int div = strategy_->divisor(step);
        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);
        int local_size = total_before[relative_rank(P)];

        MPI_Win win = create_window(in, local_size, step);
        MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOPUT, win);

        int n_blocks = size_before[relative_rank(P)].size();
        std::vector<int> rank_offset(div);

        int displacement = 0;
        for (int block = 0; block < n_blocks; block++) {
            for (int rank = 0; rank < div; ++rank) {
                int target = rank_outside_ring(P, div, off, rank);
                int b_size = size_before[target][block];

                MPI_Get(out + displacement, b_size, MPI_DOUBLE, rank, rank_offset[rank], b_size, MPI_DOUBLE, win);

                rank_offset[rank] += b_size;
                displacement += b_size;
            }
        }

        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Win_free(&win);

#ifdef DEBUG
        std::cout<< "Content of the copied matrix in rank " << rank() << " is now: "
            <<std::endl;
        for (int j = 0; j < rank_offset[gp]; j++) {
            std::cout << out[j] << ", ";
        }
        std::cout << std::endl;
#endif
    }

    void reduce(Interval& P, double* in, double* out,
            std::vector<std::vector<int>>& c_current,
            std::vector<int>& c_total_current,
            std::vector<std::vector<int>>& c_expanded,
            std::vector<int>& c_total_expanded,
            int beta, int step) override {
        int div = strategy_->divisor(step);
        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);

        int n_blocks = c_expanded[off].size();

        int target = rank_outside_ring(P, div, off, gp);
        int local_size = c_total_current[target];

        // initilize C to 0 if beta = 0 since accumulate will do additions over this array
        if (beta == 0) {
            memset(out, 0, local_size*sizeof(double));
        }

        MPI_Win win = create_window(out, local_size, step);
        MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOSTORE, win);

        int displacement = 0;
        std::vector<int> rank_offset(div);
        // go through the communication ring
        for (int block = 0; block < n_blocks; ++block) {
            for (int i = 0; i < div; ++i) {
                int target = rank_outside_ring(P, div, off, i);
                int b_size = c_current[target][block];

                MPI_Accumulate(in + displacement, b_size, MPI_DOUBLE, 
                    i, rank_offset[i], b_size, MPI_DOUBLE, MPI_SUM, win);

                displacement += b_size;
                rank_offset[i] += b_size;
            }
        }

        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Win_free(&win);
    }
};
