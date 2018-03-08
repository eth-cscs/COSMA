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

class Mapper {
public:
    Mapper(char label, int m, int n, int P, int n_steps,
         int mOffset, int nOffset,
         std::string::const_iterator patt,
         std::vector<int>::const_iterator divPatt, int rank);

    int initial_size(int rank);

    int initial_size();

    // rank -> list of ranges it owns initially
    const std::vector<Interval2D>& initial_layout(int rank) const;
    const std::vector<Interval2D>& initial_layout() const;
    std::vector<std::vector<Interval2D>> complete_layout();

    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);

    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);

    char which_matrix();

protected:
    // A, B or C
    char label_;
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

    // rank -> list of submatrices that this rank owns
    // the number of submatrices that this rank owns
    // is equal to the number of dfs steps in which
    // this matrix was divided
    std::vector<std::vector<Interval2D>> rank_to_range_;
    std::unordered_map<Interval2D, std::pair<int, int>> range_to_rank_;

    // rank -> total initial buffer size
    std::vector<int> initial_buffer_size_;

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

    void compute_sizes(Interval m, Interval n, Interval P, int step);
    void output_layout();
    void compute_range_to_rank();

};
