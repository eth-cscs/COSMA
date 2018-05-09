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

class CarmaMatrix {
public:
    CarmaMatrix(char label, const Strategy& strategy, int rank);

    int m();
    int n();

    // **********************************************
    // METHODS FROM mapper.hpp
    // **********************************************
    const int initial_size(int rank) const;

    const int initial_size() const;

    void compute_max_buffer_size(const Strategy& strategy);

    const long long max_send_buffer_size() const;
    const long long max_recv_buffer_size() const;

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
    // NEW METHODS
    // **********************************************
    double& operator[](const std::vector<double>::size_type index);

    // outputs matrix in a format:
    //      row, column, value
    // for all local elements on the current rank
    friend std::ostream& operator<<(std::ostream& os, const CarmaMatrix& mat);

    double* matrix_pointer();
    std::vector<double>& matrix();

    // copies data from matrix() to send_buffer
    void load_data();
    void unload_data();
    double* send_buffer();
    double* receive_buffer();
    double* current_matrix();
    void set_current_matrix(double* mat);
    void swap_buffers();

protected:
    // A, B or C
    char label_;
    /// local matrix
    std::vector<double> matrix_;
    /// local send buffer
    std::vector<double> send_buffer_;
    /// local receive buffer
    std::vector<double> receive_buffer_;
    /// temporary local matrix
    double* current_mat;
    /// Number of rows of the global atrix
    int m_;
    /// Number of columns of the global matrix
    int n_;
    /// Maximum number of rank in the global communicator
    size_t P_;

    int rank_;

    long long max_send_buffer_size_;
    long long max_recv_buffer_size_;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

    std::unique_ptr<Mapper> mapper_;
    std::unique_ptr<Layout> layout_;

    void compute_max_buffer_size(Interval& m, Interval& n, Interval& k, Interval& P, 
            int step, const Strategy& strategy, int rank);
};
