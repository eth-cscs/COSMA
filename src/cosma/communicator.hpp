#pragma once

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <tuple>

#include <cosma/interval.hpp>
#include <cosma/matrix.hpp>
#include <cosma/strategy.hpp>
#include <cosma/context.hpp>

#if defined(COSMA_WITH_NCCL) && defined(TILED_MM_CUDA)
#include <nccl.h>
#endif

#if defined(COSMA_WITH_NCCL) && defined(TILED_MM_ROCM)
#include <rccl.h>
#endif

namespace cosma {

// forward-declaration
// template <typename T>
// class cosma_context;

class communicator {
  public:
    communicator() = default;
    communicator(const Strategy strategy, MPI_Comm comm);
    ~communicator();

    /* In each communication step, processors are split and the communication is
     * performed. P processors are split into d groups (d = divisor in this
     * step), where each group consists of P/d processors.
     *
     * Communication rings are then created by taking 1 processor from each
     * group with the same offset within that group.
     * ------------------------------------------------------------------------------
     * Example: P = 12, d = 3:
     *    - 3 groups with 4 elements: [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
     *    - 4 communication rings: [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]
     *
     * For this reason, to each rank, we assign two numbers: (gp, offset)
     * describing which group this rank belongs to and what offset within
     * its group this rank has. The offset uniquely determines the
     * communication ring that the rank belongs to.
     * ------------------------------------------------------------------------------
     * Example: P = 12, d = 3:
     *    - rank 1 belongs to group 1 and has offset 0 within this group.
     *    - rank 4 belongs to group 0 and has offset 1 within this group.
     *    - rank 10 belongs to group 2 and has offset 2 within this group.
     * ------------------------------------------------------------------------------
     */

    /* Performs all-gather type of communication within each communiction ring.
     *
     * During the communication, ranks exchange the contents of "in" buffers.
     * After the communication, all ranks within the same communication ring
     * have exactly the same content in their "out" buffers, that contains
     * the result of the all-gather communication. This means that after
     * the communication, all the ranks within the same communication ring
     * become completely independent of each other, since they have the same
     * data regarding the communicated matrix.
     * ------------------------------------------------------------------------------
     * Example: P = 12, d = 3, first communication ring performs the following
     * communication:
     * ------------------------------------------------------------------------------
     * BEFORE COMMUNICATION:
     * rank 0:
     *      buffer "in": a1, a2, a3 (different blocks)
     * rank 4:
     *      buffer "in": b1, b2, b3
     * rank 8:
     *      buffer "in": c1, c2, c3
     * ------------------------------------------------------------------------------
     * AFTER COMMUNICATION:
     * ranks 0, 4, 8 have identical out buffer with the following content:
     *      buffer "out": a1, b1, c1, a2, b2, c2, a3, b3, c3
     * ------------------------------------------------------------------------------
     * All ranks in the same communication ring have the same number of blocks,
     * but all blocks can potentially have different sizes. The total number of
     * blocks per rank is equal to the product of all divisors in sequential
     * steps (only in sequential steps) in which this matrix was split. However,
     * not all the blocks that a rank owns are necessarily exchanged in a single
     * invocation of this function. Only blocks belonging to the current
     * submatrix are being exchanged within a single invocation of copy.
     */
    template <typename Scalar>
    void copy(Interval &P,
              Scalar *in,
              Scalar *out,
              Scalar *reshuffle_buffer,
              std::vector<std::vector<int>> &size_before,
              std::vector<int> &total_before,
              int total_after,
              int step);

    /* Performs reduce-scatter type of communication within each communiction
     * ring. This can be thought as the inverse of copy, because here all ranks
     * in the same communication ring have exactly the same number of elements
     * in their "in" buffers.
     *
     * Each rank splits the data they have into equal number of blocks and then
     * each rank reduces only a subset of blocks over all the ranks. Therefore,
     * in copy the local buffers expand after the communication, whereas here
     * the local buffers shrink because the rank wants to keep only a subset of
     * blocks.
     * ------------------------------------------------------------------------------
     * Example: P = 12, d = 3, first communication ring performs the following
     * communication:
     * ------------------------------------------------------------------------------
     * BEFORE COMMUNICATION:
     * Each rank has the same structure of data: the same number of
     * equally-sized blocks, but with possibly (almost surely) different content
     * inside blocks. This data represents the partial results of the matrix C
     * that should be reduced. Here, block a1 in rank 0 and in rank 4 can have
     * (and probably will) different content (different partial results) but the
     * size of the block a1 will be the same in all the ranks of the same
     * communication ring.
     *
     * rank 0:
     *      buffer "in": a1, b1, c1, a2, b2, c2, a3, b3, c3
     * rank 4:
     *      buffer "in": a1, b1, c1, a2, b2, c2, a3, b3, c3
     * rank 8:
     *      buffer "in": a1, b1, c1, a2, b2, c2, a3, b3, c3
     * ------------------------------------------------------------------------------
     * AFTER COMMUNICATION:
     * rank 0:
     *      buffer "in": a1, a2, a3 (summed over ranks)
     * rank 4:
     *      buffer "in": b1, b2, b3 (summed over ranks)
     * rank 8:
     *      buffer "in": c1, c2, c3 (summed over ranks)
     *
     * where each block after the communication (say block a1) is actually the
     * sum of all a1-blocks in all the ranks of this communication ring
     * ------------------------------------------------------------------------------
     * All ranks in the same communication ring have the same number of blocks,
     * but all blocks can potentially have different sizes. The total number of
     * blocks per rank is equal to the product of all divisors in sequential
     * steps (only in sequential steps) in which this matrix was split. However,
     * not all the blocks that a rank owns are necessarily exchanged in a single
     * invocation of this function. Only blocks belonging to the current
     * submatrix are being exchanged within a single invocation of reduce.
     */
    template <typename Scalar>
    void reduce(Interval &P,
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
                int step);

    template <typename Scalar>
    void overlap_comm_and_comp(cosma_context<Scalar> *ctx,
                               CosmaMatrix<Scalar> &matrixA,
                               CosmaMatrix<Scalar> &matrixB,
                               CosmaMatrix<Scalar> &matrixC,
                               Interval &m,
                               Interval &n,
                               Interval &k,
                               Interval &P,
                               size_t step,
                               Scalar alpha,
                               Scalar beta);

    // creates the graph that represents the topology of mpi communicator
    // it is "aware" of all the communications that will happen throughout
    void add_topology();

    static bool use_busy_waiting;

    // invokes MPI_Init
    static void initialize(int *argc, char ***argv);

    // rank in the initial communicator
    int rank();
    int relative_rank(Interval &P);
    int offset(Interval &P, int div);
    int group(Interval &P, int div);
    std::pair<int, int> group_and_offset(Interval &P, int div);
    int rank_inside_ring(Interval &P, int div);

    const Strategy *strategy();

    // barrier over all the ranks taking part in the multiplication
    void full_barrier();
    // barrier over the active communicator in step
    void barrier(int step);

    // communicator active in step
    MPI_Comm active_comm(int step);
#ifdef COSMA_WITH_NCCL
    // nccl communicator active in step
    ncclComm_t active_nccl_comm(int step);
#endif
    MPI_Comm full_comm();

    // size of the initial communicator
    int comm_size();

    // true if this rank is not taking part in the multiplication
    // this might happen if the total number of ranks is e.g. prime
    // or does not yield a convenient processor decomposition
    bool is_idle();

    // wrappers around MPI_Comm_free and MPI_Group_free
    static void free_comm(MPI_Comm &comm);
    static void free_group(MPI_Group &comm_group);

    // wrapper around MPI_Finalize
    static void finalize();

    static int relative_rank(Interval &P, int rank);
    static int offset(Interval &P, int div, int rank);
    static int group(Interval &P, int div, int rank);
    static std::pair<int, int> group_and_offset(Interval &P, int div, int rank);

    /*
       We split P processors into div groups of P/div processors.
     * gp from [0..(div-1)] is the id of the group of the current rank
     * offset from [0..(newP.length()-1)] is the offset of current rank inside
     its group

     We then define the communication ring of the current processor as:
     i * (P/div) + offset, where i = 0..(div-1) and offset = rank() - i *
     (P/div)
     */
    static int rank_inside_ring(Interval &P, int div, int global_rank);
    static int rank_outside_ring(Interval &P, int div, int off, int gp);

    // returns the current strategy
    const Strategy get_strategy();

  protected:
    // hierarchy of communicators used throughout the algorithm
    std::vector<MPI_Comm> comm_ring_;
    std::vector<MPI_Comm> comm_subproblem_;
    // equivalents of mpi communicators, but for nccl
#ifdef COSMA_WITH_NCCL
    std::vector<ncclComm_t> nccl_comm_ring_;
    std::vector<ncclComm_t> nccl_comm_subproblem_;
#endif
    int rank_;
    const Strategy strategy_;
    std::vector<int> step_to_comm_index_;
    MPI_Comm full_comm_ = MPI_COMM_NULL;
    int comm_size_ = 0;
    // if true then not all processors were used
    // this usually happens if given number of processors
    // cannot be decomposed nicely (e.g. if P is prime)
    bool using_reduced_comm_;
    bool is_idle_;

    void get_topology_edges(std::vector<int> &dest, std::vector<int> &weight);

    void create_communicators(MPI_Comm comm);
    // same as create just uses MPI_Comm_split instead of MPI_Comm_create
    void split_communicators(MPI_Comm comm);

    MPI_Comm create_comm_ring(MPI_Comm comm, Interval &P, int offset, int div);

    MPI_Comm create_comm_subproblem(MPI_Comm comm, Interval &P, Interval &newP);

    void free_comms();
};

} // namespace cosma
