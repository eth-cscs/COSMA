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
#include "strategy.hpp"

class Mapper {
public:
    Mapper() = default;
    Mapper(char label, int m, int n, size_t P, const Strategy& strategy, int rank);

    const size_t initial_size(int rank) const;

    const size_t initial_size() const;

    // rank -> list of ranges it owns initially
    const std::vector<Interval2D>& initial_layout(int rank) const;
    const std::vector<Interval2D>& initial_layout() const;
    std::vector<std::vector<Interval2D>>& complete_layout();

    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);

    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);

    // local_id -> (gi, gj) (for local elements on the current rank)
    // runtime: constant (pre-computed)
    const std::pair<int, int> global_coordinates(int local_index) const;

    char which_matrix();

protected:
    // A, B or C
    char label_;
    /// Number of rows of the global atrix
    int m_;
    /// Number of columns of the global matrix
    int n_;
    /// Maximum number of rank in the global communicator
    size_t P_;
    int rank_;

    // rank -> list of submatrices that this rank owns
    // the number of submatrices that this rank owns
    // is equal to the number of dfs steps in which
    // this matrix was divided
    std::vector<std::vector<Interval2D>> rank_to_range_;
    std::unordered_map<Interval2D, std::pair<int, int>> range_to_rank_;

    // rank -> total initial buffer size
    std::vector<size_t> initial_buffer_size_;

    // rank -> vector of sizes of all the ranges that this rank owns
    std::vector<std::vector<int>> range_offset_;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

private:
    // used by DFS.
    // rank -> number of submatrices fixed by the previous DFS step
    std::vector<int> skip_ranges_;

    std::set<int> row_partition_set_;
    std::set<int> col_partition_set_;
    std::vector<int> row_partition_;
    std::vector<int> col_partition_;

    std::vector<std::pair<int, int>> global_coord;

    void compute_sizes(Interval m, Interval n, Interval P, int step, const Strategy& strategy);
    void output_layout();
    void compute_range_to_rank();

    void compute_global_coord();
};
