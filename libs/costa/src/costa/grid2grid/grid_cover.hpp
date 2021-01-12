#pragma once
#include <costa/grid2grid/block.hpp>
#include <costa/grid2grid/grid2D.hpp>
#include <costa/grid2grid/interval.hpp>

#include <cassert>
#include <vector>

namespace costa {
struct interval_cover {
    int start_index = 0;
    int end_index = 0;

    interval_cover() = default;

    interval_cover(int start_idx, int end_idx)
        : start_index(start_idx)
        , end_index(end_idx) {}
};

std::ostream &operator<<(std::ostream &os, const interval_cover &other);

struct block_cover {
    interval_cover rows_cover;
    interval_cover cols_cover;

    block_cover() = default;

    block_cover(interval_cover rows_cover, interval_cover cols_cover)
        : rows_cover(rows_cover)
        , cols_cover(cols_cover) {}
};

std::vector<interval_cover>
get_decomp_cover(const std::vector<int> &decomp_blue,
                 const std::vector<int> &decomp_red);

struct grid_cover {
    std::vector<interval_cover> rows_cover;
    std::vector<interval_cover> cols_cover;

    grid_cover() = default;

    grid_cover(const grid2D &g1, const grid2D &g2) {
        rows_cover = get_decomp_cover(g1.rows_split, g2.rows_split);
        cols_cover = get_decomp_cover(g1.cols_split, g2.cols_split);
    }

    block_cover decompose_block(const block_coordinates& b) {
        return {rows_cover[b.row], cols_cover[b.col]};
    }

    template <typename T>
    block_cover decompose_block(const block<T> &b) {
        int row_index = b.coordinates.row;
        int col_index = b.coordinates.col;
        if (row_index < 0 || (size_t)row_index >= rows_cover.size() ||
            col_index < 0 || (size_t)col_index >= cols_cover.size()) {
            throw std::runtime_error(
                "Error in decompose block. Block coordinates do not belong to "
                "the grid cover.");
        }
        return {rows_cover[row_index], cols_cover[col_index]};
    }
};

// template<typename T>
// block_cover decompose_block(const grid_cover& cover, const block<T>& b);
} // namespace costa
