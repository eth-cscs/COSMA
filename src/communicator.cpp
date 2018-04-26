#include "communicator.hpp"


communicator::communicator(const Strategy& strategy, MPI_Comm comm): 
        strategy_(strategy), full_comm_(comm) {

    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &comm_size_);

    if (strategy_.topology) {
        add_topology();
    }

    create_communicators(full_comm_);
    // split_communicators(full_comm_);

    step_to_comm_index_ = std::vector<int>(strategy_.n_steps);
    int idx = 0;
    for (int i = 0; i < strategy_.n_steps; ++i) {
        step_to_comm_index_[i] = idx;
        if (strategy_.bfs_step(i)) idx++;
    }
}

communicator::~communicator() {
    free_comms();
}

void communicator::copy(Interval& P, double* in, double* out,
        std::vector<std::vector<int>>& size_before,
        std::vector<int>& total_before,
        int total_after, int step) {

    int div = strategy_.divisor(step);
    MPI_Comm subcomm = active_comm(step);

    int local_size = total_before[relative_rank(P)];

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

    if (same_size) {
        MPI_Allgather(in, local_size, MPI_DOUBLE, receive_pointer, local_size, 
                MPI_DOUBLE, subcomm);
    } else {
        // pad vectors with 0s
        MPI_Allgatherv(in, local_size, MPI_DOUBLE, receive_pointer,
                total_size.data(), dspls.data(), MPI_DOUBLE, subcomm);
    }

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

// adds vector b to vector a
void add(double* a, double* b, int n) {
    for(int i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

void communicator::reduce(Interval& P, double* LC, double* C,
        std::vector<std::vector<int>>& c_current,
        std::vector<int>& c_total_current,
        std::vector<std::vector<int>>& c_expanded,
        std::vector<int>& c_total_expanded,
        int beta, int step) {

    int div = strategy_.divisor(step);
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
    // go through the communication ring
    for (int i = 0; i < div; ++i) {
        int target = rank_outside_ring(P, div, off, i);
        recvcnts[i] = c_total_current[target];

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
    std::unique_ptr<double[]> receiving_buffer;
    double* receive_pointer;

    if (beta == 0) {
        receive_pointer = C;
    } else {
        receiving_buffer = std::unique_ptr<double[]>(new double[recvcnts[gp]]);
        receive_pointer = receiving_buffer.get();
    }

    MPI_Reduce_scatter(send_pointer, receive_pointer, recvcnts.data(), MPI_DOUBLE, MPI_SUM, subcomm);

    if (beta > 0) {
        // sum up receiving_buffer with C
        add(C, receive_pointer, recvcnts[gp]);
    }
}

// helper function for add_topology
// every communication pair represents one edge in the topology graph
// this functions finds all edges for the current rank
// weight of the edge is given by the amount of communicated data
void communicator::get_topology_edges(std::vector<int>& dest, std::vector<int>& weight) {
    int m = strategy_.m;
    int n = strategy_.n;
    int k = strategy_.k;
    Interval P(0, strategy_.P-1);
    int n_steps = strategy_.n_steps;

    for (int step = 0; step < n_steps; ++step) {
        m /= strategy_.divisor_m(step);
        n /= strategy_.divisor_n(step);
        k /= strategy_.divisor_k(step);

        if (strategy_.bfs_step(step)) {
            int div = strategy_.divisor(step);
            int partition_idx = P.partition_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div);

            for (int gp = 0; gp < div; ++gp) {
                int neighbor = P.first() + rank_outside_ring(P, div, offset, gp);
                if (neighbor == rank_) continue;
                dest.push_back(neighbor);

                int communication_size = 0;
                if (strategy_.split_n(step))
                    communication_size = m * k / newP.length();
                else if (strategy_.split_m(step))
                    communication_size = k * n / newP.length();
                else
                    communication_size = m * n / newP.length();

                weight.push_back(communication_size);
            }

            P = newP;
        }
    }
}

void communicator::add_topology() {
    int n_sources = 1;
    int source[1] = {rank_};
    std::vector<int> dest;
    std::vector<int> weight;

    get_topology_edges(dest, weight);

    int n_edges = dest.size();
    int degree = dest.size();
    int degrees[1] = {degree};

    if (n_edges >= 1) {
        MPI_Dist_graph_create(full_comm_, n_sources, source,
                degrees, dest.data(), weight.data(), MPI_INFO_NULL, true, &full_comm_);
    }
}

void communicator::initialize(int * argc, char ***argv) {
    MPI_Init(argc, argv);
}

int communicator::rank() {
    return rank_;
}

void communicator::full_barrier() {
    MPI_Barrier(full_comm_);
}

void communicator::barrier(int step) {
    int comm_index = step_to_comm_index_[step];
    MPI_Barrier(comm_ring_[comm_index]);
}

MPI_Comm communicator::active_comm(int step) {
    int comm_index = step_to_comm_index_[step];
    return comm_ring_[comm_index];
}

int communicator::comm_size() {
    return comm_size_;
}


void communicator::free_comm(MPI_Comm& comm) {
    MPI_Comm_free(&comm);
}

void communicator::free_group(MPI_Group& group) {
    MPI_Group_free(&group);
}

void communicator::finalize() {
    MPI_Finalize();
}

int communicator::relative_rank(Interval& P, int r) {
    return r - P.first();
}

int communicator::relative_rank(Interval& P) {
    return relative_rank(P, rank_);
}

int communicator::offset(Interval& P, int div, int r) {
    int subset_size = P.length() / div;
    r = relative_rank(P, r);
    int gp = r / subset_size;
    int off = r - gp * subset_size;
    return off;
}

int communicator::offset(Interval& P, int div) {
    return offset(P, div, rank_);
}

int communicator::group(Interval& P, int div, int r) {
    int subset_size = P.length() / div;
    int gp = relative_rank(P, r) / subset_size;
    return gp;
}

int communicator::group(Interval& P, int div) {
    return group(P, div, rank_);
}

std::pair<int, int> communicator::group_and_offset(Interval& P, int div, int r) {
    int subset_size = P.length() / div;
    r = relative_rank(P, r);
    int gp = r / subset_size;
    int off = r - gp * subset_size;
    return {gp, off};
}

std::pair<int, int> communicator::group_and_offset(Interval& P, int div) {
    return group_and_offset(P, div, rank_);
}

int communicator::rank_inside_ring(Interval& P, int div, int global_rank) {
    return group(P, div, global_rank);
}

int communicator::rank_inside_ring(Interval& P, int div) {
    return rank_inside_ring(P, div, rank_);
}

int communicator::rank_outside_ring(Interval& P, int div, int off, int i) {
    int subset_size = P.length() / div;
    return i * subset_size + off;
}

void communicator::split_communicators(MPI_Comm comm) {
    //MPI_Comm_group(comm, &comm_group);
    Interval P(0, strategy_.P - 1);
    // iterate through all steps and for each bfs
    // step, create a suitable subcommunicator
    for (int step = 0; step < strategy_.n_steps; ++step) {
        if (strategy_.bfs_step(step)) {
            int div = strategy_.divisor(step);
            int partition_idx = P.partition_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div, rank_);
            MPI_Comm comm_ring, comm_subproblem;
            MPI_Comm_split(comm, group, offset, &comm_subproblem);
            MPI_Comm_split(comm, offset, group, &comm_ring);
            comm_ring_.push_back(comm_ring);
            comm_subproblem_.push_back(comm_subproblem);
            comm = comm_subproblem;
            P = newP;
        }
    }
}

void communicator::create_communicators(MPI_Comm comm) {
    //MPI_Comm_group(comm, &comm_group);
    Interval P(0, strategy_.P - 1);
    // iterate through all steps and for each bfs
    // step, create a suitable subcommunicator
    for (int step = 0; step < strategy_.n_steps; ++step) {
        if (strategy_.bfs_step(step)) {
            int div = strategy_.divisor(step);
            int partition_idx = P.partition_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div, rank_);
            comm_ring_.push_back(create_comm_ring(comm, P, offset, div));
            MPI_Comm comm_subproblem = create_comm_subproblem(comm, P, newP);
            comm_subproblem_.push_back(comm_subproblem);
            comm = comm_subproblem;
            P = newP;
        }
    }
}

MPI_Comm communicator::create_comm_ring(MPI_Comm comm, Interval& P, int offset, int div) {
    MPI_Comm newcomm;
    MPI_Group subgroup;

    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);

    std::vector<int> ranks(div);
    for (int i = 0; i < div; ++i) {
        ranks[i] = rank_outside_ring(P, div, offset, i);
    }

    MPI_Group_incl(comm_group, ranks.size(), ranks.data(), &subgroup);
    MPI_Comm_create(comm, subgroup, &newcomm);

    free_group(subgroup);
    free_group(comm_group);

    return newcomm;
}

MPI_Comm communicator::create_comm_subproblem(MPI_Comm comm, Interval& P, Interval& newP) {
    MPI_Comm newcomm;
    MPI_Group subgroup;

    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);

    std::vector<int> ranks(newP.length());
    for (int i = 0; i < ranks.size(); ++i) {
        ranks[i] = relative_rank(P, newP.first() + i);
    }

    MPI_Group_incl(comm_group, ranks.size(), ranks.data(), &subgroup);
    MPI_Comm_create(comm, subgroup, &newcomm);

    free_group(subgroup);
    free_group(comm_group);

    return newcomm;
}

void communicator::free_comms() {
    for (int i = 0; i < comm_ring_.size(); ++i) {
        free_comm(comm_ring_[i]);
    }

    for (int i = 0; i < comm_subproblem_.size(); ++i) {
        free_comm(comm_subproblem_[i]);
    }
}

