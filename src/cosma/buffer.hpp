#pragma once
#include <cosma/interval.hpp>
#include <cosma/layout.hpp>
#include <cosma/mapper.hpp>
#include <cosma/strategy.hpp>
#include <cosma/context.hpp>

#ifdef COSMA_HAVE_GPU
#include <Tiled-MM/util.hpp>
#endif

#include <vector>

/*
 * This class wrapps up a vector of buffers representing single matrix (A, B or
 * C). During the algorithm, a new buffer is allocated in each parallel step in
 * which this matrix was expanded (i.e. not split). However, here we also use
 * some optimization and as soon as the number of blocks of this matrix that the
 * current rank owns reaches 1 (meaning that there are no sequential steps after
 * that step), no new allocations occur, but the sending and receiving buffers
 * keep swapping. This is possible because if there are no sequential steps
 * afterwards then after the communication matrix is expanded and the receiving
 * buffer owns everything that a sending buffer owns plus the same pieces from
 * other ranks. Therefore, after communication, we don't need the sending buffer
 * and we can reuse it to be the next receiving buffer (i.e. we can keep
 * swapping sending and receiving buffers in each parallel step in which this
 * matrix was expanded as long as there are no sequential steps left.
 */

namespace cosma {

template <typename Scalar>
class Buffer {
public:
    using scalar_t = Scalar;

    // Buffer() = default;
    Buffer();

    // with cosma_context*
    Buffer(cosma_context<Scalar>* ctxt,
           Mapper *mapper,
           Layout *layout,
           bool dry_run = false);

    // without context (using global singleton context)
    Buffer(Mapper *mapper,
           Layout *layout,
           bool dry_run = false);

    ~Buffer();

    Buffer &operator=(Buffer &) = delete;
    Buffer &operator=(Buffer &&) = default;

    // allocates all the buffers that are needed for the current matrix and the
    // current rank
    void allocate_initial_buffers(bool dry_run = false);
    void allocate_communication_buffers(bool dry_run = false);

    void free_initial_buffers(bool dry_run = false);
    void free_communication_buffers(bool dry_run = false);

    // get an array of all sizes that this matrix will require
    std::vector<size_t> get_all_buffer_sizes();

    // increases the index of the current buffer
    void advance_buffer();
    // returns the index of the current buffer
    int buffer_index();
    // sets the index of the current buffer to idx
    void set_buffer_index(int idx);
    // swaps ids of current_buffer and reduce buffer
    void swap_reduce_buffer_with(size_t buffer_idx);

    // returns the pointer to the current buffer
    scalar_t* buffer_ptr();
    const scalar_t* buffer_ptr() const;
    const size_t buffer_size() const;

    // pointer to the reshuffle buffer used when n_blocks > 1
    scalar_t *reshuffle_buffer_ptr();
    // pointer to the parallel-reduce buffer used when beta > 0
    scalar_t *reduce_buffer_ptr();
    // returns index of a buffer that is used in gemm
    // it can be either last or pre-last buffer
    // depending on the parity of #parallel steps
    // after the last sequential step.
    // (since only last two buffers keep swapping).
    int buff_index_before_gemm() const;

    // returns the initial buffer (i.e. with index 0)
    // this buffer owns the initial matrix data
    scalar_t* initial_buffer_ptr();
    const scalar_t* initial_buffer_ptr() const;
    const size_t initial_buffer_size() const;

    // we can access i-th buffer of this class with [] operator
    scalar_t* operator[](const size_t index);
    scalar_t* operator[](const size_t index) const;

    // can be A, B or C, determining the matrix
    char label_;
    // the strategy is owned by the matrix object. here only the pointer to
    // avoid the creation of the strategy again.
    const Strategy *strategy_;
    // current rank
    int rank_;

    // used to get the size of the initial buffer
    Mapper *mapper_;
    // used to get the sizes of buffers needed in each step
    Layout *layout_;

protected:
    // computes the buffer sizes that is needed for this matrix (where
    // label_="A", "B" or "C"); the length of this vector is the number of
    // different buffers that is needed.
    std::vector<size_t> compute_buffer_size();
    std::vector<size_t> compute_buffer_size(Interval &m,
                                               Interval &n,
                                               Interval &k,
                                               Interval &P,
                                               int step,
                                               int rank,
                                               scalar_t beta);

    // when the number of blocks that the current rank owns from this matrix
    // reaches 1 (meaning that there are no sequential steps left) then no new
    // buffers are allocated. From this moment, this function is invoked and it
    // follows the tree of execution of the algorithm and finds the largest two
    // buffers for this matrix that are needed. These two buffers will then be
    // reused and swapped in the whole subtree of the execution.
    void compute_max_buffer_size(Interval &m,
                                 Interval &n,
                                 Interval &k,
                                 Interval &P,
                                 int step,
                                 int rank,
                                 scalar_t beta);

    // initializes two arrays:
    // 1. n_buckets_ : this vectors gives us for each step of the algorithm the
    // number of
    //     different blocks that the current ranks owns from the current matrix.
    //     The number of blocks in step i is equal to the product of divisors in
    //     all sequential steps j > i (thus excluding step i) in which the
    //     current matrix was split.

    // 2. expanded_after_ : for each step i of the algorithm returns true/false
    // showing whether
    //     the current matrix was expanded in some of the following parallel
    //     steps (including the i-th step). The matrix is expanded in a parallel
    //     step if it is NOT split in that step.
    void compute_n_buckets();


    cosma_context<Scalar>* ctxt_;

    // computes the number of buckets in the current step
    // the number of buckets in some step i is equal to the
    // product of all divisors in sequential steps that follow step i
    // in which the current matrix was divided
    std::vector<int> n_buckets_;
    std::vector<bool> expanded_after_;

    // vector of buffers being used for the current matrix (given by label)
    // by the current rank (determined by variable rank_)
    std::vector<size_t> buffers_;
    std::vector<size_t> buff_sizes_;
    // temporary buffer used for reshuffling of data received from other ranks
    // this happens when sequential steps are present, i.e. when n_blocks > 1
    size_t reshuffle_buffer_;
    // temporary buffer used in parallel-reduce step (two-sided communication)
    // used when beta > 0 (to save current C)
    size_t reduce_buffer_;
    // pointer to the current buffer being used in the previous vector of
    // buffers
    int current_buffer_ = 0;

    // buffer used in sequential steps for reshuffling
    size_t max_reshuffle_buffer_size_ = 0;
    // buffer used in parallel reduce step, when beta == 1
    size_t max_reduce_buffer_size_ = 0;

    // computed by compute_max_buffer_size function. represent the two largest
    // buffer sizes (max_recv_buffer_size >= max_send_buffer_size);
    size_t max_send_buffer_size_ = 0;
    size_t max_recv_buffer_size_ = 0;
    size_t max_par_block_size_ = 0;
    // max size of the matrix in the base case (among all base cases)
    size_t max_base_buffer_size_ = 0;
    size_t max_send_buffer_size() const;
    size_t max_recv_buffer_size() const;

    void init_first_split_steps();
    // first seq step that splits the current matrix
    int first_seq_split_step;
    int last_first_seq_split_step;
    // first parallel step that does expands (i.e. does not split) the current
    // matrix
    int first_par_extend_step;

    // if true, memory already pinned
    bool pinned_ = false;
};

} // namespace cosma
