#pragma once
#include <cassert>
#include <fstream>
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
#include <profiler.h>

class CarmaMatrix {
public:
    CarmaMatrix(char label, int m, int n, int P, int n_steps,
        std::string::const_iterator patt,
        std::vector<int>::const_iterator divPatt, int rank);

    int m();
    int n();

    // **********************************************
    // METHODS FROM mapper.hpp
    // **********************************************
    int initial_size(int rank);

    int initial_size();

    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);

    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);

    double* matrix_pointer();

    std::vector<double>& matrix();

    char which_matrix();

    const std::vector<Interval2D>& initial_layout(int rank) const;
    const std::vector<Interval2D>& initial_layout() const;

    // **********************************************
    // METHODS FROM layout.hpp
    // **********************************************
    int offset(int rank, int dfs_bucket);
    int offset(int dfs_bucket);

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

protected:
    // A, B or C
    char label_;
    /// local matrix
    std::vector<double> matrix_;
    /// Number of rows of the global atrix
    int m_;
    /// Number of columns of the global matrix
    int n_;
    /// Maximum number of rank in the global communicator
    int P_;
    /// Number of recursive steps in the algorithm
    int n_steps_;
    /// index of the column axis related div in the division pattern
    int mOffset_;
    /// index of the row axis related div in the division pattern
    int nOffset_;
    const std::string patt_;
    const std::vector<int> divPatt_;

    int rank_;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

    std::unique_ptr<Mapper> mapper_;
    std::unique_ptr<Layout> layout_;
};
