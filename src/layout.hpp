#pragma once

//STL
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

class Layout {

public:
    Layout() = default;
    Layout(char label, int m, int n, size_t P,
           int rank, std::vector<std::vector<Interval2D>> rank_to_range);

    int size(int rank);
    int size();

    int offset(int rank, int prev_bucket);
    int offset(int prev_bucket);

    void update_buckets(Interval& P, Interval2D& range);
    int dfs_bucket(int rank);
    int dfs_bucket();
    std::vector<int> dfs_buckets(Interval& newP);
    void set_dfs_buckets(Interval& newP, std::vector<int>& pointers);

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
    char label_;

    /// Number of rows of the global atrix
    int m_;
    /// Number of columns of the global matrix
    int n_;
    /// Maximum number of rank in the global communicator
    int P_;

    int rank_;

    // rank -> list of submatrices that this rank owns
    // the number of submatrices that this rank owns
    // is equal to the number of dfs steps in which
    // this matrix was divided
    std::vector<std::vector<Interval2D>> rank_to_range_;
    // rank -> total initial buffer size
    std::vector<int> initial_size_;
    // rank -> buffer size in the current branch of the recursion
    std::vector<std::vector<int>> bucket_size_;
    std::vector<int> pointer_;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

private:
    void next(int rank);
    void next();

    void prev(int rank);
    void prev();

    std::vector<int> sizes_inside_range(Interval2D& range, int rank, int& total_size);
};
