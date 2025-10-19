#pragma once

#include <cosma/interval.hpp>
#include <cosma/strategy.hpp>

#include <costa/grid2grid/transform.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cosma {
class Mapper {
  public:
    Mapper() = default;
    Mapper(char label, const Strategy &strategy, int rank);

    Mapper(const Mapper &other);
    Mapper &operator=(const Mapper &other);
    Mapper(Mapper &&other) noexcept;
    Mapper &operator=(Mapper &&other) noexcept;

    size_t initial_size(int rank) const;

    size_t initial_size() const;

    std::vector<size_t> all_initial_sizes() const;

    // rank -> list of ranges it owns initially
    const std::vector<Interval2D> &initial_layout(int rank) const;
    const std::vector<Interval2D> &initial_layout() const;
    std::vector<std::vector<Interval2D>> &complete_layout();

    // (gi, gj) -> (local_id, rank)
    std::pair<int, int> local_coordinates(int gi, int gj);

    // (local_id, rank) -> (gi, gj)
    std::pair<int, int> global_coordinates(int local_index, int rank);

    // local_id -> (gi, gj) (for local elements on the current rank)
    std::pair<int, int> global_coordinates(int local_index);

    // returns the label of the matrix (A, B or C)
    char which_matrix();

    // get a vector of offsets of each local block
    std::vector<std::size_t> &local_blocks_offsets();

    // get a vector of local blocks
    std::vector<Interval2D> local_blocks();

    // returns a rank owning given block
    int owner(Interval2D &block);

    costa::assigned_grid2D get_layout_grid();

    int m() const;
    int n() const;
    int P() const;
    int rank() const;
    char label() const;
    const Strategy &strategy() const;

    // changes the current rank to new_rank
    // this is used when we want to reorder ranks
    // in order to minimize the communication volume
    // if matrices are initially given in a different
    // data layout
    void reorder_rank(int new_rank);

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
    const Strategy *strategy_;

    // rank -> list of submatrices that this rank owns
    // the number of submatrices that this rank owns
    // is equal to the number of sequential steps in which
    // this matrix was divided
    std::vector<std::vector<Interval2D>> rank_to_range_;
    std::unordered_map<Interval2D, std::pair<int, std::size_t>> range_to_rank_;

    // rank -> total initial buffer size
    std::vector<size_t> initial_buffer_size_;

    // rank -> vector of sizes of all the ranges that this rank owns
    std::vector<std::vector<std::size_t>> range_offset_;

    Interval mi_;
    Interval ni_;
    Interval Pi_;

  private:
    // used by sequential steps.
    // rank -> number of submatrices fixed by the previous sequential step
    std::vector<int> skip_ranges_;

    std::set<int> row_partition_set_;
    std::set<int> col_partition_set_;
    std::vector<int> row_partition_;
    std::vector<int> col_partition_;

    mutable std::mutex global_coord_mutex_;
    mutable bool global_coord_ready_{false};
    mutable std::vector<std::pair<int, int>> global_coord_;

    void compute_sizes(Interval m,
                       Interval n,
                       Interval P,
                       int step,
                       const Strategy &strategy);
    void output_layout();
    void compute_range_to_rank();

    void compute_global_coord();
};
} // namespace cosma
