#include <complex>

#include <cosma/bfloat16.hpp>
#include <cosma/communicator.hpp>
#include <cosma/one_sided_communicator.hpp>
#include <cosma/two_sided_communicator.hpp>

#if defined(COSMA_HAVE_GPU) && defined(COSMA_WITH_NCCL)
#include <cosma/gpu/nccl_utils.hpp>
#endif

namespace cosma {
bool communicator::use_busy_waiting = true;

communicator::communicator(const Strategy strategy, MPI_Comm comm)
    : strategy_(strategy) {

    use_busy_waiting = strategy_.use_busy_waiting;

    MPI_Comm_rank(comm, &rank_);
    // rank_ = reordered_rank(rank_);
    MPI_Comm_size(comm, &comm_size_);
    // check if the reordered rank belongs
    // to this communicator
    assert(rank_ < comm_size_);
    using_reduced_comm_ = comm_size_ != strategy.P;
    is_idle_ = rank_ >= strategy.P;

    if (using_reduced_comm_) {
        MPI_Group group;
        MPI_Comm_group(comm, &group);
        std::vector<int> exclude_ranks;
        for (int i = strategy.P; i < comm_size_; ++i) {
            // exclude_ranks.push_back(reordered_rank(i));
            exclude_ranks.push_back(i);
        }

        MPI_Group reduced_group;

        MPI_Group_excl(
            group, exclude_ranks.size(), exclude_ranks.data(), &reduced_group);
        MPI_Comm_create_group(comm, reduced_group, 0, &full_comm_);

        MPI_Group_free(&group);
        MPI_Group_free(&reduced_group);
    } else {
        // this communicator has to be duplicated as it's being cached.
        // The user might deallocate the comm as it's allocated outside of COSMA
        // for this reason, we have to ensure that we duplicate the comm
        // before it's cached in the COSMA context
        MPI_Comm_dup(comm, &full_comm_);
        // full_comm_ = comm;
    }

    if (is_idle_) {
        return;
    }

    if (strategy_.topology) {
        add_topology();
    }

    create_communicators(full_comm_);
    // split_communicators(full_comm_);

    step_to_comm_index_ = std::vector<int>(strategy_.n_steps());
    int idx = 0;
    for (int i = 0; i < strategy_.n_steps(); ++i) {
        step_to_comm_index_[i] = idx;
        if (strategy_.parallel_step(i))
            idx++;
    }
}

communicator::~communicator() {
    if (!is_idle_) {
        free_comms();
    }
}

bool communicator::is_idle() { return is_idle_; }

// helper function for add_topology
// every communication pair represents one edge in the topology graph
// this functions finds all edges for the current rank
// weight of the edge is given by the amount of communicated data
void communicator::get_topology_edges(std::vector<int> &dest,
                                      std::vector<int> &weight) {
    int m = strategy_.m;
    int n = strategy_.n;
    int k = strategy_.k;
    Interval P(0, strategy_.P - 1);
    int n_steps = strategy_.n_steps();

    for (int step = 0; step < n_steps; ++step) {
        m /= strategy_.divisor_m(step);
        n /= strategy_.divisor_n(step);
        k /= strategy_.divisor_k(step);

        if (strategy_.parallel_step(step)) {
            int div = strategy_.divisor(step);
            int partition_idx = P.subinterval_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div);

            for (int gp = 0; gp < div; ++gp) {
                int neighbor =
                    P.first() + rank_outside_ring(P, div, offset, gp);
                if (neighbor == rank_)
                    continue;
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
        MPI_Dist_graph_create(full_comm_,
                              n_sources,
                              source,
                              degrees,
                              dest.data(),
                              weight.data(),
                              MPI_INFO_NULL,
                              true,
                              &full_comm_);
    }
}

void communicator::initialize(int *argc, char ***argv) { MPI_Init(argc, argv); }

int communicator::rank() { return rank_; }

void communicator::full_barrier() { MPI_Barrier(full_comm_); }

void communicator::barrier(int step) {
    int comm_index = step_to_comm_index_[step];
    MPI_Barrier(comm_ring_[comm_index]);
}

MPI_Comm communicator::full_comm() { return full_comm_; }

MPI_Comm communicator::active_comm(int step) {
    int comm_index = step_to_comm_index_[step];
    return comm_ring_[comm_index];
}

#ifdef COSMA_WITH_NCCL
ncclComm_t communicator::active_nccl_comm(int step) {
    int comm_index = step_to_comm_index_[step];
    return nccl_comm_ring_[comm_index];
}
#endif

int communicator::comm_size() { return comm_size_; }

void communicator::free_comm(MPI_Comm &comm) {
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Comm_free(&comm);
    }
}

void communicator::free_group(MPI_Group &group) { MPI_Group_free(&group); }

void communicator::finalize() { MPI_Finalize(); }

int communicator::relative_rank(Interval &P, int r) { return r - P.first(); }

int communicator::relative_rank(Interval &P) { return relative_rank(P, rank_); }

int communicator::offset(Interval &P, int div, int r) {
    return P.subinterval_offset(div, r);
}

int communicator::offset(Interval &P, int div) { return offset(P, div, rank_); }

int communicator::group(Interval &P, int div, int r) {
    return P.subinterval_index(div, r);
}

int communicator::group(Interval &P, int div) { return group(P, div, rank_); }

std::pair<int, int>
communicator::group_and_offset(Interval &P, int div, int r) {
    return P.locate_in_subinterval(div, r);
}

std::pair<int, int> communicator::group_and_offset(Interval &P, int div) {
    return group_and_offset(P, div, rank_);
}

int communicator::rank_inside_ring(Interval &P, int div, int global_rank) {
    return group(P, div, global_rank);
}

int communicator::rank_inside_ring(Interval &P, int div) {
    return rank_inside_ring(P, div, rank_);
}

int communicator::rank_outside_ring(Interval &P, int div, int off, int i) {
    return P.locate_in_interval(div, i, off);
}

void communicator::split_communicators(MPI_Comm comm) {
    // MPI_Comm_group(comm, &comm_group);
    Interval P(0, strategy_.P - 1);
    // iterate through all steps and for each parallel
    // step, create a suitable subcommunicator
    for (int step = 0; step < strategy_.n_steps(); ++step) {
        if (strategy_.parallel_step(step)) {
            int div = strategy_.divisor(step);
            int partition_idx = P.subinterval_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div, rank_);
            MPI_Comm comm_ring, comm_subproblem;
            MPI_Comm_split(comm, group, offset, &comm_subproblem);
            MPI_Comm_split(comm, offset, group, &comm_ring);

            comm_ring_.push_back(comm_ring);
            comm_subproblem_.push_back(comm_subproblem);

#ifdef COSMA_WITH_NCCL
            nccl_comm_ring_.push_back(gpu::mpi_to_nccl_comm(comm_ring_.back()));
            nccl_comm_subproblem_.push_back(
                gpu::mpi_to_nccl_comm(comm_subproblem_.back()));
#endif

            comm = comm_subproblem;
            P = newP;
        }
    }
}

MPI_Comm create_comm(MPI_Comm &comm, std::vector<int> &ranks) {
    MPI_Comm newcomm;
    MPI_Group subgroup;

    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);

    MPI_Group_incl(comm_group, ranks.size(), ranks.data(), &subgroup);
    MPI_Comm_create_group(comm, subgroup, 0, &newcomm);

    communicator::free_group(subgroup);
    communicator::free_group(comm_group);

    return newcomm;
}

void communicator::create_communicators(MPI_Comm comm) {
    // MPI_Comm_group(comm, &comm_group);
    Interval P(0, strategy_.P - 1);

    // iterate through all steps and for each parallel
    // step, create a suitable subcommunicator
    for (int step = 0; step < strategy_.n_steps(); ++step) {
        if (strategy_.parallel_step(step)) {
            int div = strategy_.divisor(step);
            int partition_idx = P.subinterval_index(div, rank_);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = group_and_offset(P, div, rank_);

            comm_ring_.emplace_back(create_comm_ring(comm, P, offset, div));
            comm_subproblem_.emplace_back(
                create_comm_subproblem(comm, P, newP));

#ifdef COSMA_WITH_NCCL
            nccl_comm_ring_.emplace_back(
                gpu::mpi_to_nccl_comm(comm_ring_.back()));
            nccl_comm_subproblem_.emplace_back(
                gpu::mpi_to_nccl_comm(comm_subproblem_.back()));
#endif

            comm = comm_subproblem_.back();
            P = newP;
        }
    }
}

MPI_Comm communicator::create_comm_ring(MPI_Comm comm,
                                        Interval &P,
                                        int offset,
                                        int div) {
    std::vector<int> ranks(div);
    for (int i = 0; i < div; ++i) {
        ranks[i] = rank_outside_ring(P, div, offset, i);
    }

    return create_comm(comm, ranks);
}

MPI_Comm communicator::create_comm_subproblem(MPI_Comm comm,
                                              Interval &P,
                                              Interval &newP) {
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
    for (int i = comm_subproblem_.size() - 1; i >= 0; --i) {
        free_comm(comm_subproblem_[i]);
#ifdef COSMA_WITH_NCCL
        gpu::free_nccl_comm(nccl_comm_subproblem_[i]);
#endif
    }
    for (int i = comm_ring_.size() - 1; i >= 0; --i) {
        free_comm(comm_ring_[i]);
#ifdef COSMA_WITH_NCCL
        gpu::free_nccl_comm(nccl_comm_ring_[i]);
#endif
    }
    // if (using_reduced_comm_) {
    free_comm(full_comm_);
    full_comm_ = MPI_COMM_NULL;
    //}
}

template <typename Scalar>
void communicator::copy(Interval &P,
                        Scalar *in,
                        Scalar *out,
                        Scalar *reshuffle_buffer,
                        std::vector<std::vector<int>> &size_before,
                        std::vector<int> &total_before,
                        int total_after,
                        int step) {
    MPI_Comm comm = active_comm(step);
    two_sided_communicator::copy(comm,
                                 rank(),
                                 strategy_.divisor(step),
                                 P,
                                 in,
                                 out,
                                 reshuffle_buffer,
                                 size_before,
                                 total_before,
                                 total_after);
}

template <typename Scalar>
void communicator::reduce(Interval &P,
                          Scalar *in,
                          Scalar *out,
                          Scalar *reshuffle_buffer,
                          Scalar *reduce_buffer,
                          std::vector<std::vector<int>> &c_current,
                          std::vector<int> &c_total_current,
                          std::vector<std::vector<int>> &c_expanded,
                          std::vector<int> &c_total_expanded,
                          Scalar alpha,
                          Scalar beta,
                          int step) {
    MPI_Comm comm = active_comm(step);
    two_sided_communicator::reduce(comm,
                                   rank(),
                                   strategy_.divisor(step),
                                   P,
                                   in,  // LC
                                   out, // C
                                   reshuffle_buffer,
                                   reduce_buffer,
                                   c_current,
                                   c_total_current,
                                   c_expanded,
                                   c_total_expanded,
                                   beta);
}

template <typename Scalar>
void communicator::overlap_comm_and_comp(cosma_context<Scalar> *ctx,
                                         CosmaMatrix<Scalar> &matrixA,
                                         CosmaMatrix<Scalar> &matrixB,
                                         CosmaMatrix<Scalar> &matrixC,
                                         Interval &m,
                                         Interval &n,
                                         Interval &k,
                                         Interval &P,
                                         size_t step,
                                         Scalar alpha,
                                         Scalar beta) {
    MPI_Comm comm = active_comm(step);
    one_sided_communicator::overlap_comm_and_comp(ctx,
                                                  comm,
                                                  rank(),
                                                  strategy_,
                                                  matrixA,
                                                  matrixB,
                                                  matrixC,
                                                  m,
                                                  n,
                                                  k,
                                                  P,
                                                  step,
                                                  alpha,
                                                  beta);
}

const Strategy communicator::get_strategy() { return strategy_; }

// Explicit instantiations for `copy`
//
template void
communicator::copy<double>(Interval &P,
                           double *in,
                           double *out,
                           double *reshuffle_buffer,
                           std::vector<std::vector<int>> &size_before,
                           std::vector<int> &total_before,
                           int total_after,
                           int step);

template void
communicator::copy<float>(Interval &P,
                          float *in,
                          float *out,
                          float *reshuffle_buffer,
                          std::vector<std::vector<int>> &size_before,
                          std::vector<int> &total_before,
                          int total_after,
                          int step);

template void communicator::copy<std::complex<float>>(
    Interval &P,
    std::complex<float> *in,
    std::complex<float> *out,
    std::complex<float> *reshuffle_buffer,
    std::vector<std::vector<int>> &size_before,
    std::vector<int> &total_before,
    int total_after,
    int step);

template void communicator::copy<std::complex<double>>(
    Interval &P,
    std::complex<double> *in,
    std::complex<double> *out,
    std::complex<double> *reshuffle_buffer,
    std::vector<std::vector<int>> &size_before,
    std::vector<int> &total_before,
    int total_after,
    int step);

template void
communicator::copy<bfloat16>(Interval &P,
                             bfloat16 *in,
                             bfloat16 *out,
                             bfloat16 *reshuffle_buffer,
                             std::vector<std::vector<int>> &size_before,
                             std::vector<int> &total_before,
                             int total_after,
                             int step);

// Explicit instantiations for `reduce`
//
template void
communicator::reduce<float>(Interval &P,
                            float *in,
                            float *out,
                            float *reshuffle_buffer,
                            float *reduce_buffer,
                            std::vector<std::vector<int>> &c_current,
                            std::vector<int> &c_total_current,
                            std::vector<std::vector<int>> &c_expanded,
                            std::vector<int> &c_total_expanded,
                            float alpha,
                            float beta,
                            int step);

template void
communicator::reduce<double>(Interval &P,
                             double *in,
                             double *out,
                             double *reshuffle_buffer,
                             double *reduce_buffer,
                             std::vector<std::vector<int>> &c_current,
                             std::vector<int> &c_total_current,
                             std::vector<std::vector<int>> &c_expanded,
                             std::vector<int> &c_total_expanded,
                             double alpha,
                             double beta,
                             int step);

template void communicator::reduce<std::complex<float>>(
    Interval &P,
    std::complex<float> *in,
    std::complex<float> *out,
    std::complex<float> *reshuffle_buffer,
    std::complex<float> *reduce_buffer,
    std::vector<std::vector<int>> &c_current,
    std::vector<int> &c_total_current,
    std::vector<std::vector<int>> &c_expanded,
    std::vector<int> &c_total_expanded,
    std::complex<float> alpha,
    std::complex<float> beta,
    int step);

template void communicator::reduce<std::complex<double>>(
    Interval &P,
    std::complex<double> *in,
    std::complex<double> *out,
    std::complex<double> *reshuffle_buffer,
    std::complex<double> *reduce_buffer,
    std::vector<std::vector<int>> &c_current,
    std::vector<int> &c_total_current,
    std::vector<std::vector<int>> &c_expanded,
    std::vector<int> &c_total_expanded,
    std::complex<double> alpha,
    std::complex<double> beta,
    int step);

template void
communicator::reduce<bfloat16>(Interval &P,
                               bfloat16 *in,
                               bfloat16 *out,
                               bfloat16 *reshuffle_buffer,
                               bfloat16 *reduce_buffer,
                               std::vector<std::vector<int>> &c_current,
                               std::vector<int> &c_total_current,
                               std::vector<std::vector<int>> &c_expanded,
                               std::vector<int> &c_total_expanded,
                               bfloat16 alpha,
                               bfloat16 beta,
                               int step);

// Explicit instantiations for `overlap_comm_and_comp`
//
template void
communicator::overlap_comm_and_comp<double>(cosma_context<double> *ctx,
                                            CosmaMatrix<double> &matrixA,
                                            CosmaMatrix<double> &matrixB,
                                            CosmaMatrix<double> &matrixC,
                                            Interval &m,
                                            Interval &n,
                                            Interval &k,
                                            Interval &P,
                                            size_t step,
                                            double alpha,
                                            double beta);
template void
communicator::overlap_comm_and_comp<float>(cosma_context<float> *ctx,
                                           CosmaMatrix<float> &matrixA,
                                           CosmaMatrix<float> &matrixB,
                                           CosmaMatrix<float> &matrixC,
                                           Interval &m,
                                           Interval &n,
                                           Interval &k,
                                           Interval &P,
                                           size_t step,
                                           float alpha,
                                           float beta);

template void communicator::overlap_comm_and_comp<std::complex<float>>(
    cosma_context<std::complex<float>> *ctx,
    CosmaMatrix<std::complex<float>> &matrixA,
    CosmaMatrix<std::complex<float>> &matrixB,
    CosmaMatrix<std::complex<float>> &matrixC,
    Interval &m,
    Interval &n,
    Interval &k,
    Interval &P,
    size_t step,
    std::complex<float> alpha,
    std::complex<float> beta);

template void communicator::overlap_comm_and_comp<std::complex<double>>(
    cosma_context<std::complex<double>> *ctx,
    CosmaMatrix<std::complex<double>> &matrixA,
    CosmaMatrix<std::complex<double>> &matrixB,
    CosmaMatrix<std::complex<double>> &matrixC,
    Interval &m,
    Interval &n,
    Interval &k,
    Interval &P,
    size_t step,
    std::complex<double> alpha,
    std::complex<double> beta);

template void
communicator::overlap_comm_and_comp<bfloat16>(cosma_context<bfloat16> *ctx,
                                              CosmaMatrix<bfloat16> &matrixA,
                                              CosmaMatrix<bfloat16> &matrixB,
                                              CosmaMatrix<bfloat16> &matrixC,
                                              Interval &m,
                                              Interval &n,
                                              Interval &k,
                                              Interval &P,
                                              size_t step,
                                              bfloat16 alpha,
                                              bfloat16 beta);

} // namespace cosma
