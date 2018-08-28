#pragma once
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include "interval.hpp"
#include <unordered_map>
#include <algorithm>
#include "mapper.hpp"
#include "layout.hpp"
#include <semiprof.hpp>
#include "strategy.hpp"
#include "buffer.hpp"

class CosmaMatrix {
public:
    CosmaMatrix(char label, const Strategy& strategy, int rank);

    int m();
    int n();
    char label();

    // **********************************************
    // METHODS FROM mapper.hpp
    // **********************************************
    const int initial_size(int rank) const;
    const int initial_size() const;

    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);
    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);
    // local_id -> (gi, gj) for local elements on the current rank
    // runtime: constant (pre-computed)
    const std::pair<int, int> global_coordinates(int local_index) const;

    char which_matrix();

    const std::vector<Interval2D>& initial_layout(int rank) const;
    const std::vector<Interval2D>& initial_layout() const;

    // **********************************************
    // METHODS FROM layout.hpp
    // **********************************************
    int shift(int rank, int dfs_bucket);
    int shift(int dfs_bucket);
    void unshift(int offset);

    void update_buckets(Interval& P, Interval2D& range);
    int dfs_bucket(int rank);
    int dfs_bucket();
    std::vector<int> dfs_buckets(Interval& newP);
    void set_dfs_buckets(Interval& newP, std::vector<int>& pointers);

    int size(int rank);
    int size();

    void buffers_before_expansion(Interval& P, Interval2D& range,
            std::vector<std::vector<int>>& size_per_rank,
            std::vector<int>& total_size_per_rank);

    void buffers_after_expansion(Interval& P, Interval& newP,
            std::vector<std::vector<int>>& size_per_rank,
            std::vector<int>& total_size_per_rank,
            std::vector<std::vector<int>>& new_size,
            std::vector<int>& new_total);

    void set_sizes(Interval& newP, std::vector<std::vector<int>>& size_per_rank, int offset);
    void set_sizes(Interval& newP, std::vector<std::vector<int>>& size_per_rank);
    void set_sizes(int rank, std::vector<int>& sizes, int start);

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
    double* buffer_ptr();

    std::vector<double, mpi_allocator<double>>& buffer();
    const std::vector<double, mpi_allocator<double>>& buffer() const;

    // **********************************************
    // NEW METHODS
    // **********************************************
    double& operator[](const std::vector<double>::size_type index);
    double operator[](const std::vector<double>::size_type index) const;

    // outputs matrix in a format:
    //      row, column, value
    // for all local elements on the current rank
    friend std::ostream& operator<<(std::ostream& os, const CosmaMatrix& mat);

    double* matrix_pointer();
    std::vector<double, mpi_allocator<double>>& matrix();
    const std::vector<double, mpi_allocator<double>>& matrix() const;

    // pointer to send buffer
    //double* buffer_ptr();
    //std::vector<double, mpi_allocator<double>>& buffer();
    // pointer to current matrix (send buffer)
    double* current_matrix();
    void set_current_matrix(double* mat);

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

    const Strategy& strategy_;

    /// temporary local matrix
    double* current_mat;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

    Mapper mapper_;
    Layout layout_;
    Buffer buffer_;

    std::vector<double, mpi_allocator<double>> dummy_vector;
};
