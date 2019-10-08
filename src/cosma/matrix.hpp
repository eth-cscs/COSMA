#pragma once

#include <cosma/buffer.hpp>
#include <cosma/context.hpp>
#include <cosma/interval.hpp>
#include <cosma/layout.hpp>
#include <cosma/mapper.hpp>
#include <cosma/strategy.hpp>

#include <grid2grid/transform.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace cosma {

template <typename Scalar>
class CosmaMatrix {
  public:
    using scalar_t = Scalar;
    using buffer_t = Buffer<scalar_t>;

    // using a pointer to cosma_context
    CosmaMatrix(cosma_context<Scalar>* ctxt,
                char label,
                const Strategy &strategy,
                int rank,
                bool dry_run = false);
    CosmaMatrix(cosma_context<Scalar>* ctxt,
                Mapper&& mapper, int rank, bool dry_run = false);

    // using a custom context
    CosmaMatrix(std::unique_ptr<cosma_context<Scalar>>& ctxt,
                char label,
                const Strategy &strategy,
                int rank,
                bool dry_run = false);
    CosmaMatrix(std::unique_ptr<cosma_context<Scalar>>& ctxt,
                Mapper&& mapper, int rank, bool dry_run = false);

    // using global (singleton) context
    CosmaMatrix(char label,
                const Strategy &strategy,
                int rank,
                bool dry_run = false);
    CosmaMatrix(Mapper&& mapper, int rank, bool dry_run = false);

    int m();
    int n();
    char label();

    // **********************************************
    // METHODS FROM mapper.hpp
    // **********************************************
    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);
    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);
    // local_id -> (gi, gj) for local elements on the current rank
    std::pair<int, int> global_coordinates(int local_index);

    char which_matrix();

    const std::vector<Interval2D> &initial_layout(int rank) const;
    const std::vector<Interval2D> &initial_layout() const;

    // **********************************************
    // METHODS FROM layout.hpp
    // **********************************************
    int shift(int rank, int seq_bucket);
    int shift(int seq_bucket);
    void unshift(int offset);

    void update_buckets(Interval &P, Interval2D &range);
    int seq_bucket(int rank);
    int seq_bucket();
    std::vector<int> seq_buckets(Interval &newP);
    void set_seq_buckets(Interval &newP, std::vector<int> &pointers);

    int size(int rank);
    int size();

    void buffers_before_expansion(Interval &P,
                                  Interval2D &range,
                                  std::vector<std::vector<int>> &size_per_rank,
                                  std::vector<int> &total_size_per_rank);

    void buffers_after_expansion(Interval &P,
                                 Interval &newP,
                                 std::vector<std::vector<int>> &size_per_rank,
                                 std::vector<int> &total_size_per_rank,
                                 std::vector<std::vector<int>> &new_size,
                                 std::vector<int> &new_total);

    void set_sizes(Interval &newP,
                   std::vector<std::vector<int>> &size_per_rank,
                   int offset);
    void set_sizes(Interval &newP,
                   std::vector<std::vector<int>> &size_per_rank);
    void set_sizes(int rank, std::vector<int> &sizes, int start);

    // **********************************************
    // METHODS FROM buffer.hpp
    // **********************************************
    // prepares next buffer
    void advance_buffer();
    // returns the current buffer id
    int buffer_index();
    // sets the current buffer to idx
    void set_buffer_index(int idx);
    // returns the pointer to the current buffer
    scalar_t *buffer_ptr();
    // returns the pointer to the reshuffle buffer
    // that is used when n_blocks > 1 (i.e. when sequential steps are present)
    // as a temporary buffer in which the data is reshuffled.
    scalar_t *reshuffle_buffer_ptr();
    // pointer to the reduce buffer that is used as a
    // temporary buffer in parallel-reduce (two-sided) communicator
    // in case when beta > 0 in that step
    scalar_t *reduce_buffer_ptr();

    // **********************************************
    // NEW METHODS
    // **********************************************
    scalar_t &operator[](const typename std::vector<scalar_t>::size_type index);
    scalar_t operator[](const typename std::vector<scalar_t>::size_type index) const;

    // outputs matrix in a format:
    //      row, column, value
    // for all local elements on the current rank
    template <typename Scalar_>
    friend std::ostream &operator<<(std::ostream &os,
                                    const CosmaMatrix<Scalar_> &mat);

    // get a pointer to the initial/final data
    scalar_t *matrix_pointer();
    const scalar_t *matrix_pointer() const;
    size_t matrix_size() const;
    size_t matrix_size(int rank) const;

    // pointer to send buffer
    // scalar_t* buffer_ptr();
    // std::vector<scalar_t, mpi_allocator<scalar_t>>& buffer();
    // pointer to current matrix (send buffer)
    scalar_t *current_matrix();

    // this should be invoked after all allocations are finished
    // it will query the memory pool for the current buffers
    void initialize();

    void set_current_matrix(scalar_t *mat);

    grid2grid::grid_layout<scalar_t> get_grid_layout();

    void allocate_communication_buffers();
    void free_communication_buffers();

    cosma_context<scalar_t>* get_context();

    int rank() const;

  protected:
    cosma_context<scalar_t>* ctxt_;
    // mapper containing information
    // about the global grid (data layout)
    Mapper mapper_;
    // current rank
    int rank_;
    // strategy
    const Strategy &strategy_;

    // A, B or C
    char label_;
    /// Number of rows of the global matrix
    int m_;
    /// Number of columns of the global matrix
    int n_;
    /// Maximum number of rank in the global communicator
    size_t P_;

    /// temporary local matrix
    size_t current_mat_id;
    scalar_t *current_mat;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

    Layout layout_;
    buffer_t buffer_;
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &os, CosmaMatrix<Scalar> &mat) {
    for (auto local = 0; local < mat.matrix_size(); ++local) {
        auto value = mat[local];
        int row, col;
        std::tie(row, col) = mat.global_coordinates(local);
        os << row << " " << col << " " << value << std::endl;
    }
    return os;
}

} // namespace cosma
