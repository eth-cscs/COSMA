#pragma once
#include "interval.hpp"
#include "strategy.hpp"
#include <vector>
#include "mapper.hpp"
#include "layout.hpp"
#include "mpi_allocator.hpp"
#include "communicator.hpp"

#ifdef COSMA_HAVE_GPU
//#include "./gpu/cuda_allocator.hpp"
#include "./gpu/device_vector.hpp"
#endif

/*
 * This class wrapps up a vector of buffers representing single matrix (A, B or C).
 * During the algorithm, a new buffer is allocated in each BFS step in which 
 * this matrix was expanded (i.e. not split). However, here we also use some optimization
 * and as soon as the number of blocks of this matrix that the current rank owns
 * reaches 1 (meaning that there are no DFS steps after that step), no new allocations
 * occur, but the sending and receiving buffers keep swapping. This is possible
 * because if there are no DFS steps afterwards then after the communication
 * matrix is expanded and the receiving buffer owns everything that a sending buffer owns
 * plus the same pieces from other ranks. Therefore, after communication, we don't need 
 * the sending buffer and we can reuse it to be the next receiving buffer (i.e. we can keep
 * swapping sending and receiving buffers in each BFS step in which this matrix was expanded
 * as long as there are no DFS steps left.
 */

class Buffer {
public:
    Buffer() = default;
    Buffer(char label, const Strategy& strategy, int rank, Mapper* mapper, Layout* layout);

    // allocates all the buffers that are needed for the current matrix and the current rank
    void initialize_buffers();

    // increases the index of the current buffer
    void advance_buffer();
    // returns the index of the current buffer
    int buffer_index();
    // sets the index of the current buffer to idx
    void set_buffer_index(int idx);

    // returns the pointer to the current buffer
    double* buffer_ptr();
    // pointer to the reshuffle buffer used when n_blocks > 1
    double* reshuffle_buffer_ptr();
    // pointer to the BFS-reduce buffer used when beta > 0
    double* reduce_buffer_ptr();
    // returns a reference to the current buffer
    std::vector<double, mpi_allocator<double>>& buffer();
    const std::vector<double, mpi_allocator<double>>& buffer() const;

    // returns the initial buffer (i.e. with index 0)
    // this buffer owns the initial matrix data
    double* initial_buffer_ptr();
    std::vector<double, mpi_allocator<double>>& initial_buffer();
    const std::vector<double, mpi_allocator<double>>& initial_buffer() const;

    // we can access i-th buffer of this class with [] operator
    std::vector<double, mpi_allocator<double>>& operator[](const std::vector<double, mpi_allocator<double>>::size_type index);
    std::vector<double, mpi_allocator<double>> operator[](const std::vector<double, mpi_allocator<double>>::size_type index) const;

    // can be A, B or C, determining the matrix
    char label_;
    // the strategy is owned by the matrix object. here only the pointer to 
    // avoid the creation of the strategy again.
    const Strategy* strategy_;
    // current rank
    int rank_;

    // used to get the size of the initial buffer
    Mapper* mapper_;
    // used to get the sizes of buffers needed in each step
    Layout* layout_;

#ifdef COSMA_HAVE_GPU
    static const int tile_size_m = 4200;
    static const int tile_size_n = 4200;
    static const int tile_size_k = 4200;
    static const int n_streams = 3;

    device_vector<double> device_buffer_;
    double* device_buffer_ptr();
    // buffer used for overlapping tiled gemm
    double* intermediate_buffer_;
    double* intermediate_buffer_ptr();
#endif

protected:
    // computes the buffer sizes that is needed for this matrix (where label_="A", "B" or "C");
    // the length of this vector is the number of different buffers that is needed.
    // this function just triggers the recursive function with the same name
    std::vector<long long> compute_buffer_size();
    std::vector<long long> compute_buffer_size(Interval& m, Interval& n, Interval& k, Interval& P, 
        int step, int rank, double beta);

    // when the number of blocks that the current rank owns from this matrix reaches 1
    // (meaning that there are no DFS steps left) then no new buffers are allocated. 
    // From this moment, this function is invoked and it follows the tree of execution 
    // of the algorithm and finds the largest two buffers for this matrix that are needed. 
    // These two buffers will then be reused and swapped in the whole subtree of the execution.
    void compute_max_buffer_size(Interval& m, Interval& n, Interval& k, Interval& P, 
        int step, int rank, double beta);

    // initializes two arrays:
    // 1. n_buckets_ : this vectors gives us for each step of the algorithm the number of 
    //     different blocks that the current ranks owns from the current matrix.
    //     The number of blocks in step i is equal to the product of divisors in all DFS steps j > i
    //     (thus excluding step i) in which the current matrix was split.

    // 2. expanded_after_ : for each step i of the algorithm returns true/false showing whether 
    //     the current matrix was expanded in some of the following BFS steps (including the 
    //     i-th step). The matrix is expanded in a BFS step if it is NOT split in that step.
    void compute_n_buckets();

    // computes the number of buckets in the current step
    // the number of buckets in some step i is equal to the
    // product of all divisors in DFS steps that follow step i
    // in which the current matrix was divided
    std::vector<int> n_buckets_;
    std::vector<bool> expanded_after_;

    // vector of buffers being used for the current matrix (given by label) 
    // by the current rank (determined by variable rank_)
    std::vector<std::vector<double, mpi_allocator<double>>> buffers_;
    // temporary buffer used for reshuffling of data received from other ranks
    // this happens when DFS steps are present, i.e. when n_blocks > 1
    std::unique_ptr<double[]> reshuffle_buffer_;
    // temporary buffer used in BFS-reduce step (two-sided communication)
    // used when beta > 0 (to save current C)
    std::unique_ptr<double[]> reduce_buffer_;
    // pointer to the current buffer being used in the previous vector of buffers
    int current_buffer_;

    // buffer used in DFS steps for reshuffling
    long long max_reshuffle_buffer_size_;
    // buffer used in BFS reduce step, when beta == 1
    long long max_reduce_buffer_size_;

    // computed by compute_max_buffer_size function. represent the two largest buffer sizes 
    // (max_recv_buffer_size >= max_send_buffer_size);
    long long max_send_buffer_size_;
    long long max_recv_buffer_size_;
    // max size of the matrix in the base case (among all base cases)
    long long max_base_buffer_size_;
    const long long max_send_buffer_size() const;
    const long long max_recv_buffer_size() const;
};
