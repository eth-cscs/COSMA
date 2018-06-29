#include "communicator.hpp"

class one_sided_communicator: public communicator {
public:
    one_sided_communicator(const Strategy* strategy, MPI_Comm comm): communicator(strategy, comm) {}

    void copy(Interval& P, double* in, double* out,
            std::vector<std::vector<int>>& size_before,
            std::vector<int>& total_before,
            int total_after, int step) override {
        int div = strategy_->divisor(step);
        MPI_Comm subcomm = active_comm(step);

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

        int sum = 0;
        std::vector<int> total_size(div);
        std::vector<int> dspls(div);
        int off = offset(P, div);

        std::vector<int> subgroup(div);
        bool same_size = true;
        int max_size = 0;

        for (int i = 0; i < div; ++i) {
            int target = rank_outside_ring(P, div, off, i);
            int temp_size = total_before[target];
            dspls[i] = sum;
            sum += temp_size;
            total_size[i] = temp_size;
            same_size &= temp_size == local_size;
            max_size = std::max(max_size, temp_size);
        }

        int n_buckets = size_before[relative_rank(P)].size();
        double* receive_pointer;
        std::unique_ptr<double[]> receiving_buffer;

        if (n_buckets > 1) {
            receiving_buffer = std::unique_ptr<double[]>(new double[total_after]);
            receive_pointer = receiving_buffer.get();
        } else {
            receive_pointer = out;
        }

        for (int i = 0; i < div; ++i) {
            MPI_Get(receive_pointer + dspls[i], total_size[i], MPI_DOUBLE, i, 0, total_size[i], MPI_DOUBLE, win);
        }

        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        MPI_Win_free(&win);

        if (n_buckets > 1) {
            int index = 0;
            std::vector<int> bucket_offset(div);
            // order all first DFS parts of all groups first and so on..
            for (int bucket = 0; bucket < n_buckets; bucket++) {
                for (int rank = 0; rank < div; rank++) {
                    int target = rank_outside_ring(P, div, off, rank);
                    int dsp = dspls[rank] + bucket_offset[rank];
                    int b_size = size_before[target][bucket];
                    std::copy(receiving_buffer.get() + dsp, receiving_buffer.get() + dsp + b_size, out + index);
                    index += b_size;
                    bucket_offset[rank] += b_size;
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
            std::vector<std::vector<int>>& c_current,
            std::vector<int>& c_total_current,
            std::vector<std::vector<int>>& c_expanded,
            std::vector<int>& c_total_expanded,
            int beta, int step) override {
        std::cout << "Invoking reduce of one sided backend" << std::endl;
        int div = strategy_->divisor(step);
        MPI_Comm subcomm = active_comm(step);

        std::vector<int> subgroup(div);

        int gp, off;
        std::tie(gp, off) = group_and_offset(P, div);

        // reorder the elements as:
        // first all buckets that should be sent to rank 0 then all buckets for rank 1 and so on...
        int n_buckets = c_expanded[off].size();
        std::vector<int> bucket_offset(n_buckets);
        std::unique_ptr<double[]> send_buffer;
        double* send_pointer;

        int sum = 0;
        for (int i = 0; i < n_buckets; ++i) {
            bucket_offset[i] = sum;
            sum += c_expanded[off][i];
        }

        std::vector<int> recvcnts(div);

        if (n_buckets > 1) {
            send_buffer = std::unique_ptr<double[]>(new double[c_total_expanded[off]]);
            send_pointer = send_buffer.get();
        } else {
            send_pointer = LC;
        }

        int index = 0;
        std::vector<int> dspls(div);
        sum = 0;
        // go through the communication ring
        for (int i = 0; i < div; ++i) {
            int target = rank_outside_ring(P, div, off, i);
            int size = c_total_current[target];
            recvcnts[i] = size;
            dspls[i] = sum;
            sum += size;

            if (n_buckets > 1) {
                for (int bucket = 0; bucket < n_buckets; ++bucket) {
                    int b_offset = bucket_offset[bucket];
                    int b_size = c_current[target][bucket];
                    std::copy(LC + b_offset, LC + b_offset + b_size, send_buffer.get() + index);
                    index += b_size;
                    bucket_offset[bucket] += b_size;
                }
            }
        }

        int local_size = recvcnts[gp];

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
        /*
        MPI_Win_create_dynamic(MPI_INFO_NULL, subcomm, &win);
        MPI_Barrier(subcomm);
        MPI_Win_attach(win, receive_pointer, local_size*sizeof(double));
        MPI_Barrier(subcomm);
        */
        MPI_Win_fence(MPI_MODE_NOPRECEDE + MPI_MODE_NOSTORE, win);

        for (int i = 0; i < div; ++i) {
            MPI_Accumulate(send_pointer + dspls[i], recvcnts[i], MPI_DOUBLE, 
                i, 0, recvcnts[i], MPI_DOUBLE, MPI_SUM, win);
        }

        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
        // MPI_Win_detach(win, receive_pointer);
        MPI_Win_free(&win);
    }

    // no need to synchronize anything since copy and reduce are blocking calls
    void synchronize() {}
};
