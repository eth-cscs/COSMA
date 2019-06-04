#include "communicator.hpp"

namespace cosma {
bool communicator::use_busy_waiting = true;

communicator::communicator(const Strategy* strategy, MPI_Comm comm):
        strategy_(strategy), full_comm_(comm) {
    use_busy_waiting = strategy_->use_busy_waiting;
    MPI_Group group;

    MPI_Comm_rank(full_comm_, &rank_);
    MPI_Comm_size(full_comm_, &comm_size_);
    using_reduced_comm_ = comm_size_ != strategy->P;
    is_idle_ = rank_ >= strategy->P;

    if (using_reduced_comm_) {
        MPI_Comm_group(comm, &group);
        std::vector<int> exclude_ranks;
        for (int i = strategy->P; i < comm_size_; ++i) {
            exclude_ranks.push_back(i);
        }

        MPI_Group_excl(group, exclude_ranks.size(), exclude_ranks.data(), &full_comm_group_);
        MPI_Comm_create_group(comm, full_comm_group_, 0, &full_comm_);

        MPI_Group_free(&group);
    }

    if (is_idle_) {
        return;
    }

    if (strategy_->topology) {
        add_topology();
    }

    create_communicators(full_comm_);
    // split_communicators(full_comm_);

    step_to_comm_index_ = std::vector<int>(strategy_->n_steps);
    int idx = 0;
    for (int i = 0; i < strategy_->n_steps; ++i) {
        step_to_comm_index_[i] = idx;
        if (strategy_->parallel_step(i)) idx++;
    }
}

communicator::~communicator() {
    if (!is_idle_) {
        free_comms();
    }
}

bool communicator::is_idle() {
    return is_idle_;
}

// helper function for add_topology
// every communication pair represents one edge in the topology graph
// this functions finds all edges for the current rank
// weight of the edge is given by the amount of communicated data
void communicator::get_topology_edges(std::vector<int>& dest, std::vector<int>& weight) {
    int m = strategy_->m;
    int n = strategy_->n;
    int k = strategy_->k;
    Interval P(0, strategy_->P-1);
    int n_steps = strategy_->n_steps;

    for (int step = 0; step < n_steps; ++step) {
        m /= strategy_->divisor_m(step);
        n /= strategy_->divisor_n(step);
        k /= strategy_->divisor_k(step);

        if (strategy_->parallel_step(step)) {
            int div = strategy_->divisor(step);
            int partition_idx = P.subinterval_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div);

            for (int gp = 0; gp < div; ++gp) {
                int neighbor = P.first() + rank_outside_ring(P, div, offset, gp);
                if (neighbor == rank_) continue;
                dest.push_back(neighbor);

                int communication_size = 0;
                if (strategy_->split_n(step))
                    communication_size = m * k / newP.length();
                else if (strategy_->split_m(step))
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
    return P.subinterval_offset(div, r);
}

int communicator::offset(Interval& P, int div) {
    return offset(P, div, rank_);
}

int communicator::group(Interval& P, int div, int r) {
    return P.subinterval_index(div, r);
}

int communicator::group(Interval& P, int div) {
    return group(P, div, rank_);
}

std::pair<int, int> communicator::group_and_offset(Interval& P, int div, int r) {
    return P.locate_in_subinterval(div, r);
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
    return P.locate_in_interval(div, i, off);
}

void communicator::split_communicators(MPI_Comm comm) {
    //MPI_Comm_group(comm, &comm_group);
    Interval P(0, strategy_->P - 1);
    // iterate through all steps and for each parallel
    // step, create a suitable subcommunicator
    for (int step = 0; step < strategy_->n_steps; ++step) {
        if (strategy_->parallel_step(step)) {
            int div = strategy_->divisor(step);
            int partition_idx = P.subinterval_index(div, rank_);
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
    Interval P(0, strategy_->P - 1);
    // iterate through all steps and for each parallel
    // step, create a suitable subcommunicator
    for (int step = 0; step < strategy_->n_steps; ++step) {
        if (strategy_->parallel_step(step)) {
            int div = strategy_->divisor(step);
            int partition_idx = P.subinterval_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div, rank_);

            MPI_Group ring_group;
            MPI_Comm ring_comm;
            std::tie(ring_group, ring_comm) = create_comm_ring(comm, P, offset, div);
            comm_ring_.push_back(ring_comm);
            comm_ring_group_.push_back(ring_group);

            MPI_Group subproblem_group;
            MPI_Comm subproblem_comm;
            std::tie(subproblem_group, subproblem_comm) = create_comm_subproblem(comm, P, newP);
            comm_subproblem_.push_back(subproblem_comm);
            comm_subproblem_group_.push_back(subproblem_group);

            comm = subproblem_comm;
            P = newP;
        }
    }
}

std::tuple<MPI_Group, MPI_Comm> communicator::create_comm_ring(MPI_Comm comm, Interval& P,
        int offset, int div) {
    MPI_Comm newcomm;
    MPI_Group subgroup;

    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);

    std::vector<int> ranks(div);
    for (int i = 0; i < div; ++i) {
        ranks[i] = rank_outside_ring(P, div, offset, i);
    }

    MPI_Group_incl(comm_group, ranks.size(), ranks.data(), &subgroup);
    MPI_Comm_create_group(comm, subgroup, 0, &newcomm);

    // free_group(subgroup);
    free_group(comm_group);

    return {subgroup, newcomm};
}

std::tuple<MPI_Group, MPI_Comm> communicator::create_comm_subproblem(MPI_Comm comm, Interval& P, 
        Interval& newP) {
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

    // free_group(subgroup);
    free_group(comm_group);

    return {subgroup, newcomm};
}

void communicator::free_comms() {
    if (using_reduced_comm_) {
        free_group(full_comm_group_);
        free_comm(full_comm_);
    }
    for (int i = 0; i < comm_ring_.size(); ++i) {
        free_group(comm_ring_group_[i]);
        free_comm(comm_ring_[i]);
    }

    for (int i = 0; i < comm_subproblem_.size(); ++i) {
        free_group(comm_subproblem_group_[i]);
        free_comm(comm_subproblem_[i]);
    }
}
}
