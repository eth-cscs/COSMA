#pragma once

#include <cosma/buffer.hpp>
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
    using mpi_buffer_t = typename buffer_t::mpi_buffer_t;

    CosmaMatrix(char label,
                const Strategy &strategy,
                int rank,
                bool dry_run = false);

    int m();
    int n();
    char label();

    // **********************************************
    // METHODS FROM mapper.hpp
    // **********************************************
    int initial_size(int rank) const;
    int initial_size() const;

    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);
    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);
    // local_id -> (gi, gj) for local elements on the current rank
    // runtime: constant (pre-computed)
    const std::pair<int, int> global_coordinates(int local_index) const;

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

    mpi_buffer_t &buffer();
    const mpi_buffer_t &buffer() const;

    // **********************************************
    // NEW METHODS
    // **********************************************
    scalar_t &operator[](const typename std::vector<scalar_t>::size_type index);
    scalar_t
    operator[](const typename std::vector<scalar_t>::size_type index) const;

    // outputs matrix in a format:
    //      row, column, value
    // for all local elements on the current rank
    template <typename Scalar_>
    friend std::ostream &operator<<(std::ostream &os,
                                    const CosmaMatrix<Scalar_> &mat);

    scalar_t *matrix_pointer();
    mpi_buffer_t &matrix();
    const mpi_buffer_t &matrix() const;

    // pointer to send buffer
    // scalar_t* buffer_ptr();
    // std::vector<scalar_t, mpi_allocator<scalar_t>>& buffer();
    // pointer to current matrix (send buffer)
    scalar_t *current_matrix();
    void set_current_matrix(scalar_t *mat);

    grid2grid::grid_layout<scalar_t> get_grid_layout();

  protected:
    // A, B or C
    char label_;
    /// Number of rows of the global matrix
    int m_;
    /// Number of columns of the global matrix
    int n_;
    /// Maximum number of rank in the global communicator
    size_t P_;

    int rank_;

    const Strategy &strategy_;

    /// temporary local matrix
    scalar_t *current_mat;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

    Mapper mapper_;
    Layout layout_;
    buffer_t buffer_;

    mpi_buffer_t dummy_vector;
};

template <typename Scalar>
std::ostream &operator<<(std::ostream &os, const CosmaMatrix<Scalar> &mat) {
    for (auto local = 0; local < mat.initial_size(); ++local) {
        auto value = mat[local];
        int row, col;
        std::tie(row, col) = mat.global_coordinates(local);
        os << row << " " << col << " " << value << std::endl;
    }
    return os;
}

} // namespace cosma
