#pragma once
#include <costa/grid2grid/profiler.hpp>
#include <costa/grid2grid/grid_layout.hpp>
#include <costa/grid2grid/grid_cover.hpp>
#include <costa/grid2grid/communication_data.hpp>
#include <algorithm>

namespace costa {
namespace utils {
template <typename T>
std::vector<message<T>> decompose_block(const block<T> &b,
                                        grid_cover &g_cover,
                                        const assigned_grid2D &g,
                                        const T alpha, const T beta) {
    // std::cout << "decomposing block " << b << std::endl;
    block_cover b_cover = g_cover.decompose_block(b);

    int row_first = b_cover.rows_cover.start_index;
    int row_last = b_cover.rows_cover.end_index;

    int col_first = b_cover.cols_cover.start_index;
    int col_last = b_cover.cols_cover.end_index;

    std::vector<message<T>> decomposed_blocks;

    int row_start = b.rows_interval.start;
    // use start of the interval to get the rank and the end of the interval
    // to get the block which has to be sent
    // skip the last element
    for (int i = row_first; i < row_last; ++i) {
        int row_end = std::min(g.grid().rows_split[i + 1], b.rows_interval.end);

        int col_start = b.cols_interval.start;
        for (int j = col_first; j < col_last; ++j) {
            // use i, j to find out the rank
            int rank = g.owner(i, j);
            // std::cout << "owner of block " << i << ", " << j << " is " <<
            // rank << std::endl;

            // use i+1 and j+1 to find out the block
            int col_end =
                std::min(g.grid().cols_split[j + 1], b.cols_interval.end);

            // get pointer to this block of data based on the internal local
            // layout
            block<T> subblock =
                b.subblock({row_start, row_end}, {col_start, col_end});

            assert(subblock.non_empty());
            // if non empty, add this block
            if (subblock.non_empty()) {
                // std::cout << "for rank " << rank << ", adding subblock: " <<
                // subblock << std::endl; std::cout << "owner of " << subblock
                // << " is " << rank << std::endl;
                decomposed_blocks.push_back({subblock, rank});
                decomposed_blocks.back().alpha = alpha;
                decomposed_blocks.back().beta = beta;
            }

            col_start = col_end;
        }
        row_start = row_end;
    }
    return decomposed_blocks;
}

template <typename T>
std::vector<message<T>> decompose_blocks(const grid_layout<T> &init_layout,
                                         const grid_layout<T> &final_layout,
                                         const T alpha, const T beta,
                                         int tag = 0) {
    PE(transform_decompose);
    grid_cover g_overlap(init_layout.grid.grid(), final_layout.grid.grid());

    std::vector<message<T>> messages;

    for (int i = 0; i < init_layout.blocks.num_blocks(); ++i) {
        // std::cout << "decomposing block " << i << " out of " <<
        // init_layout.blocks.num_blocks() << std::endl;
        auto blk = init_layout.blocks.get_block(i);
        blk.tag = tag;
        assert(blk.non_empty());
        std::vector<message<T>> decomposed =
            decompose_block(blk, g_overlap, final_layout.grid, alpha, beta);
        messages.insert(messages.end(), decomposed.begin(), decomposed.end());
    }
    PL();
    return messages;
}

std::unordered_map<int, int> rank_to_comm_vol_for_block(
        const assigned_grid2D& g_init,
        const block_coordinates &b_coord,
        grid_cover &g_cover,
        const assigned_grid2D& g_final) {
    // std::cout << "decomposing block " << b << std::endl;
    block_cover b_cover = g_cover.decompose_block(b_coord);

    int row_first = b_cover.rows_cover.start_index;
    int row_last = b_cover.rows_cover.end_index;

    int col_first = b_cover.cols_cover.start_index;
    int col_last = b_cover.cols_cover.end_index;

    auto rows_interval = g_init.rows_interval(b_coord.row);
    auto cols_interval = g_init.cols_interval(b_coord.col);

    std::unordered_map<int, int> comm_vol;

    int row_start = rows_interval.start;
    // use start of the interval to get the rank and the end of the interval
    // to get the block which has to be sent
    // skip the last element
    for (int i = row_first; i < row_last; ++i) {
        int row_end = std::min(g_final.grid().rows_split[i + 1], rows_interval.end);

        int col_start = cols_interval.start;
        for (int j = col_first; j < col_last; ++j) {
            // use i, j to find out the rank
            int rank = g_final.owner(i, j);
            // std::cout << "owner of block " << i << ", " << j << " is " <<
            // rank << std::endl;

            // use i+1 and j+1 to find out the block
            int col_end =
                std::min(g_final.grid().cols_split[j + 1], cols_interval.end);

            int size = (row_end - row_start) * (col_end - col_start);
            // if non empty, add this block
            if (size) {
                comm_vol[rank] += size;
            }

            col_start = col_end;
        }
        row_start = row_end;
    }
    return comm_vol;
}

template <typename T>
void merge_messages(std::vector<message<T>> &messages) {
    std::sort(messages.begin(), messages.end());
}

template <typename T>
communication_data<T> prepare_to_send(const grid_layout<T> &init_layout,
                                      const grid_layout<T> &final_layout,
                                      int rank,
                                      const T alpha, const T beta) {
    // in case ranks were reordered to minimize the communication
    // this might not be the identity function
    // if (rank == 0) {
    //     std::cout << "prepare to send: changing rank to " << init_layout.reordered_rank(rank) << std::endl;
    // }
    // rank = init_layout.reordered_rank(rank);
    std::vector<message<T>> messages =
        decompose_blocks(init_layout, final_layout, alpha, beta);
    merge_messages(messages);
    return communication_data<T>(messages, rank, std::max(final_layout.num_ranks(), init_layout.num_ranks()));
}

template <typename T>
communication_data<T> prepare_to_send(
                                      std::vector<layout_ref<T>>& from,
                                      std::vector<layout_ref<T>>& to,
                                      int rank,
                                      const T* alpha, const T* beta) {
    std::vector<message<T>> messages;
    int n_ranks = 0;

    for (unsigned i = 0u; i < from.size(); ++i) {
        auto& init_layout = from[i].get();
        auto& final_layout = to[i].get();
        auto decomposed_blocks = decompose_blocks(init_layout, final_layout, alpha[i], beta[i], i);
        messages.insert(messages.end(), decomposed_blocks.begin(), decomposed_blocks.end());
        n_ranks = std::max(n_ranks, std::max(final_layout.num_ranks(), init_layout.num_ranks()));
    }
    merge_messages(messages);
    return communication_data<T>(messages, rank, n_ranks);
}

template <typename T>
communication_data<T> prepare_to_recv(const grid_layout<T> &final_layout,
                                      const grid_layout<T> &init_layout,
                                      int rank,
                                      const T alpha, const T beta) {
    std::vector<message<T>> messages =
        decompose_blocks(final_layout, init_layout, alpha, beta);
    merge_messages(messages);
    return communication_data<T>(messages, rank, std::max(init_layout.num_ranks(), final_layout.num_ranks()));
}

template <typename T>
communication_data<T> prepare_to_recv(
                                      std::vector<layout_ref<T>>& to,
                                      std::vector<layout_ref<T>>& from,
                                      int rank,
                                      const T* alpha, const T* beta) {
    std::vector<message<T>> messages;
    int n_ranks = 0;

    for (unsigned i = 0u; i < from.size(); ++i) {
        auto& init_layout = from[i].get();
        auto& final_layout = to[i].get();
        auto decomposed_blocks = decompose_blocks(final_layout, init_layout, alpha[i], beta[i], i);
        messages.insert(messages.end(), decomposed_blocks.begin(), decomposed_blocks.end());
        n_ranks = std::max(n_ranks, std::max(init_layout.num_ranks(), final_layout.num_ranks()));
    }
    merge_messages(messages);
    return communication_data<T>(messages, rank, n_ranks);
}
} // namespace utils
} // namespace costa

