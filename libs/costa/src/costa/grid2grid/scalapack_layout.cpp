#include <costa/grid2grid/scalapack_layout.hpp>

namespace costa {
namespace scalapack {

std::ostream &operator<<(std::ostream &os, const int_pair &other) {
    os << "[" << other.row << ", " << other.col << "]";
    return os;
}
rank_grid_coord
rank_to_grid(int rank, rank_decomposition grid_dim, ordering grid_ord) {
    if (rank < 0 || rank >= grid_dim.n_total()) {
        throw std::runtime_error(
            "Error in rank_to_grid: rank does not belong to the grid.");
    }

    if (grid_ord == ordering::column_major) {
        int ld = grid_dim.row;
        return {rank % ld, rank / ld};
    } else {
        int ld = grid_dim.col;
        return {rank / ld, rank % ld};
    }
}

rank_grid_coord rank_to_grid(int rank,
                             rank_decomposition grid_dim,
                             ordering grid_ord,
                             rank_grid_coord src) {
    if (rank < 0 || rank >= grid_dim.n_total()) {
        throw std::runtime_error(
            "Error in rank_to_grid: rank does not belong to the grid.");
    }

    rank_grid_coord r_coord = rank_to_grid(rank, grid_dim, grid_ord);
    r_coord = (r_coord + src) % grid_dim;
    return r_coord;
}

int rank_from_grid(rank_grid_coord grid_coord,
                   rank_decomposition grid_dim,
                   ordering grid_ord) {
    if (grid_coord.row < 0 || grid_coord.row >= grid_dim.row ||
        grid_coord.col < 0 || grid_coord.col >= grid_dim.col) {
        throw std::runtime_error(
            "Error in rank_from_grid: rank coordinates do not belong \
    to the rank grid.");
    }

    if (grid_ord == ordering::column_major) {
        int ld = grid_dim.row;
        return grid_coord.col * ld + grid_coord.row;
    } else {
        int ld = grid_dim.col;
        return grid_coord.row * ld + grid_coord.col;
    }
}

std::tuple<int, int> local_coordinate(int glob_coord,
                                      int block_dimension,
                                      int p_block_dimension,
                                      int mat_dim) {
    int idx_block = glob_coord / block_dimension;
    int idx_in_block = glob_coord % block_dimension;
    int idx_block_proc = idx_block / p_block_dimension;
    int owner = idx_block % p_block_dimension;

    return std::make_tuple(idx_block_proc * block_dimension + idx_in_block, owner);
}

// global->local coordinates
local_grid_coord local_coordinates(matrix_grid mat_grid,
                                   rank_decomposition rank_grid,
                                   elem_grid_coord global_coord) {
    int row, p_row;
    std::tie<int, int>(row, p_row) =
        local_coordinate(global_coord.row,
                         mat_grid.block_dimension.row,
                         rank_grid.row,
                         mat_grid.matrix_dimension.row);
    int col, p_col;
    std::tie<int, int>(col, p_col) =
        local_coordinate(global_coord.col,
                         mat_grid.block_dimension.col,
                         rank_grid.col,
                         mat_grid.matrix_dimension.row);
    return {{row, col}, {p_row, p_col}};
}

// local->global coordinates
elem_grid_coord global_coordinates(matrix_grid mat_grid,
                                   rank_decomposition rank_grid,
                                   local_grid_coord local_coord) {
    int li = local_coord.el_coord.row;
    int lj = local_coord.el_coord.col;

    int my_row_rank = local_coord.rank_coord.row;
    int my_col_rank = local_coord.rank_coord.col;

    int num_row_ranks = rank_grid.row;
    int num_col_ranks = rank_grid.col;

    int mb = mat_grid.block_dimension.row;
    int nb = mat_grid.block_dimension.col;

    int gi = ((li / mb) * num_row_ranks + my_row_rank) * mb + (li % mb);
    int gj = ((lj / nb) * num_col_ranks + my_col_rank) * nb + (lj % nb);

    if (gi < 0 || gi > mat_grid.matrix_dimension.row || gj < 0 ||
        gj >= mat_grid.matrix_dimension.col) {
        // std::cout << "coordinates (" << gi << ", " << gj << ") should belong
        // to ("
        //     << mat_grid.matrix_dimension.row << ", " <<
        //     mat_grid.matrix_dimension.col << ")" << std::endl;
        // throw std::runtime_error("ERROR in scalapack::global_coordinates,
        // values out of range.");
        return {-1, -1};
    }
    return {gi, gj};
}

local_blocks get_local_blocks(matrix_grid mat_grid,
                              rank_decomposition r_grid,
                              rank_grid_coord rank_coord) {
    auto b_dim = mat_grid.block_dimension;
    auto m_dim = mat_grid.matrix_dimension;

    int n_blocks_row = (int)std::ceil(1.0 * m_dim.row / b_dim.row);
    int n_blocks_col = (int)std::ceil(1.0 * m_dim.col / b_dim.col);

    int n_owning_blocks_row =
        n_blocks_row / r_grid.row +
        (rank_coord.row < n_blocks_row % r_grid.row ? 1 : 0);
    int n_owning_blocks_col =
        n_blocks_col / r_grid.col +
        (rank_coord.col < n_blocks_col % r_grid.col ? 1 : 0);

    return {n_owning_blocks_row, n_owning_blocks_col, b_dim, rank_coord};
}

size_t local_size(int rank, data_layout &layout) {
    matrix_grid mat_grid(layout.matrix_dimension, layout.block_dimension);
    rank_grid_coord rank_coord =
        rank_to_grid(rank, layout.rank_grid, layout.rank_grid_ordering);
    local_blocks loc_blocks =
        get_local_blocks(mat_grid, layout.rank_grid, rank_coord);

    return loc_blocks.size_with_padding();
}
} // namespace scalapack

inline std::vector<int> line_split(int begin, int end, int blk_len) {
    int len = end - begin;
    int rem = blk_len - begin % blk_len;

    std::vector<int> splits{0};

    if (rem >= len) {
        splits.push_back(len);
        return splits;
    }

    if (rem != 0) {
        splits.push_back(rem);
    }

    int num_blocks = (len - rem) / blk_len;
    for (int i = 0; i < num_blocks; ++i) {
        splits.push_back(splits.back() + blk_len);
    }

    if (splits.back() != len) {
        splits.push_back(len);
    }

    return splits;
}
template <typename T>
grid_layout<T> get_scalapack_layout(
    int lld,                                    // local leading dim
    scalapack::matrix_dim matrix_shape,         // global matrix size
    scalapack::elem_grid_coord submatrix_begin, // start of submatrix (from 1)
    scalapack::matrix_dim submatrix_shape,      // dim of submatrix
    scalapack::block_dim blk_shape,             // block dimension
    scalapack::rank_decomposition ranks_grid,
    scalapack::ordering ranks_grid_ordering,
    scalapack::rank_grid_coord ranks_grid_src_coord,
    T *ptr,
    const int rank) {

    assert(submatrix_begin.row >= 1);
    assert(submatrix_begin.col >= 1);

    submatrix_begin.row--;
    submatrix_begin.col--;

    std::vector<int> rows_split =
        line_split(submatrix_begin.row,
                   submatrix_begin.row + submatrix_shape.row,
                   blk_shape.row);
    std::vector<int> cols_split =
        line_split(submatrix_begin.col,
                   submatrix_begin.col + submatrix_shape.col,
                   blk_shape.col);

    int blk_grid_rows = static_cast<int>(rows_split.size() - 1);
    int blk_grid_cols = static_cast<int>(cols_split.size() - 1);

    std::vector<std::vector<int>> owners(blk_grid_rows,
                                         std::vector<int>(blk_grid_cols));
    std::vector<block<T>> loc_blocks;
    loc_blocks.reserve(blk_grid_rows * blk_grid_cols);

    // The begin block grid coordinates of the matrix block which is inside or
    // is split by the submatrix.
    //
    int border_blk_row_begin = submatrix_begin.row / blk_shape.row;
    int border_blk_col_begin = submatrix_begin.col / blk_shape.col;

    scalapack::rank_grid_coord submatrix_rank_grid_src_coord{
        (border_blk_row_begin % ranks_grid.row + ranks_grid_src_coord.row) %
            ranks_grid.row,
        (border_blk_col_begin % ranks_grid.col + ranks_grid_src_coord.col) %
            ranks_grid.col};

    // Iterate over the grid of blocks of the submatrix.
    //
    for (int j = 0; j < blk_grid_cols; ++j) {
        int rank_col =
            (j % ranks_grid.col + submatrix_rank_grid_src_coord.col) %
            ranks_grid.col;
        for (int i = 0; i < blk_grid_rows; ++i) {
            int rank_row =
                (i % ranks_grid.row + submatrix_rank_grid_src_coord.row) %
                ranks_grid.row;

            // The rank to which the block belongs
            //
            owners[i][j] = rank_from_grid(
                {rank_row, rank_col}, ranks_grid, ranks_grid_ordering);

            // If block belongs to current rank
            //
            if (owners[i][j] == rank) {
                // Coordinates of the (border) block within this rank.
                //
                int blk_loc_row = (border_blk_row_begin + i) / ranks_grid.row;
                int blk_loc_col = (border_blk_col_begin + j) / ranks_grid.col;

                // The begin coordinates of the sub-block within the local block
                // in this process.
                //
                int subblk_loc_row = submatrix_begin.row + rows_split[i] -
                                     (border_blk_row_begin + i) * blk_shape.row;
                int subblk_loc_col = submatrix_begin.col + cols_split[j] -
                                     (border_blk_col_begin + j) * blk_shape.col;

                int data_offset =
                    blk_loc_row * blk_shape.row + subblk_loc_row +
                    lld * (blk_loc_col * blk_shape.col + subblk_loc_col);

                loc_blocks.push_back(
                    {{rows_split[i], rows_split[i + 1]}, // rows
                     {cols_split[j], cols_split[j + 1]}, // cols
                     {i, j},                             // blk coords
                     ptr + data_offset,
                     lld});
            }
        }
    }

    grid2D grid(std::move(rows_split), std::move(cols_split));
    assigned_grid2D assigned_grid(
        std::move(grid), std::move(owners), ranks_grid.n_total());
    local_blocks<T> local_memory(std::move(loc_blocks));
    grid_layout<T> layout(std::move(assigned_grid), std::move(local_memory));
    return layout;
}

assigned_grid2D get_scalapack_grid(
    scalapack::matrix_dim matrix_shape,         // global matrix size
    scalapack::elem_grid_coord submatrix_begin, // start of submatrix (from 1)
    scalapack::matrix_dim submatrix_shape,      // dim of submatrix
    scalapack::block_dim blk_shape,             // block dimension
    scalapack::rank_decomposition ranks_grid,
    scalapack::ordering ranks_grid_ordering,
    scalapack::rank_grid_coord ranks_grid_src_coord) {

    assert(submatrix_begin.row >= 1);
    assert(submatrix_begin.col >= 1);

    submatrix_begin.row--;
    submatrix_begin.col--;

    std::vector<int> rows_split =
        line_split(submatrix_begin.row,
                   submatrix_begin.row + submatrix_shape.row,
                   blk_shape.row);
    std::vector<int> cols_split =
        line_split(submatrix_begin.col,
                   submatrix_begin.col + submatrix_shape.col,
                   blk_shape.col);

    int blk_grid_rows = static_cast<int>(rows_split.size() - 1);
    int blk_grid_cols = static_cast<int>(cols_split.size() - 1);

    std::vector<std::vector<int>> owners(blk_grid_rows,
                                         std::vector<int>(blk_grid_cols));

    // The begin block grid coordinates of the matrix block which is inside or
    // is split by the submatrix.
    //
    int border_blk_row_begin = submatrix_begin.row / blk_shape.row;
    int border_blk_col_begin = submatrix_begin.col / blk_shape.col;

    scalapack::rank_grid_coord submatrix_rank_grid_src_coord{
        (border_blk_row_begin % ranks_grid.row + ranks_grid_src_coord.row) %
            ranks_grid.row,
        (border_blk_col_begin % ranks_grid.col + ranks_grid_src_coord.col) %
            ranks_grid.col};

    // Iterate over the grid of blocks of the submatrix.
    //
    for (int j = 0; j < blk_grid_cols; ++j) {
        int rank_col =
            (j % ranks_grid.col + submatrix_rank_grid_src_coord.col) %
            ranks_grid.col;
        for (int i = 0; i < blk_grid_rows; ++i) {
            int rank_row =
                (i % ranks_grid.row + submatrix_rank_grid_src_coord.row) %
                ranks_grid.row;

            // The rank to which the block belongs
            //
            owners[i][j] = rank_from_grid(
                {rank_row, rank_col}, ranks_grid, ranks_grid_ordering);
        }
    }

    grid2D grid(std::move(rows_split), std::move(cols_split));
    assigned_grid2D assigned_grid(
        std::move(grid), std::move(owners), ranks_grid.n_total());
    return assigned_grid;
}

template <typename T>
grid_layout<T>
get_scalapack_layout(int lld,             // local leading dim
                   scalapack::matrix_dim m_dim,    // global matrix size
                   scalapack::elem_grid_coord ij,  // start of submatrix
                   scalapack::matrix_dim subm_dim, // dim of submatrix
                   scalapack::block_dim b_dim,     // block dimension
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   const T *ptr,
                   const int rank) {

    return get_scalapack_layout(lld,
                              m_dim,
                              ij,
                              subm_dim,
                              b_dim,
                              r_grid,
                              rank_grid_ordering,
                              rank_src,
                              const_cast<T *>(ptr),
                              rank);
}

template <typename T>
grid_layout<T> get_scalapack_layout(scalapack::matrix_dim m_dim,
                                  scalapack::block_dim b_dim,
                                  scalapack::rank_decomposition r_grid,
                                  scalapack::ordering rank_grid_ordering,
                                  T *ptr,
                                  int rank) {
    // std::cout << "I AM RANK " << rank << std::endl;
    int n_blocks_row = (int)std::ceil(1.0 * m_dim.row / b_dim.row);
    int n_blocks_col = (int)std::ceil(1.0 * m_dim.col / b_dim.col);

    scalapack::rank_grid_coord rank_coord =
        rank_to_grid(rank, r_grid, rank_grid_ordering);

    int n_owning_blocks_row =
        n_blocks_row / r_grid.row +
        (rank_coord.row < (n_blocks_row % r_grid.row) ? 1 : 0);

    int stride = n_owning_blocks_row * b_dim.row;

    return get_scalapack_layout(stride, // local leading dim
                              m_dim,  // global matrix size
                              {1, 1}, // start of submatrix
                              m_dim,  // dim of submatrix
                              b_dim,  // block dimension
                              r_grid,
                              rank_grid_ordering,
                              {0, 0},
                              ptr,
                              rank);
}

template <typename T>
grid_layout<T>
get_scalapack_layout(scalapack::data_layout &layout, T *ptr, int rank) {
    return get_scalapack_layout<T>(layout.matrix_dimension,
                                 layout.block_dimension,
                                 layout.rank_grid,
                                 layout.rank_grid_ordering,
                                 ptr,
                                 rank);
}
// template instantiation for get_scalapack_layout
template grid_layout<float>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   float *ptr,
                   const int rank);

template grid_layout<double>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   double *ptr,
                   const int rank);

template grid_layout<std::complex<float>>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   std::complex<float> *ptr,
                   const int rank);

template grid_layout<std::complex<double>>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   std::complex<double> *ptr,
                   const int rank);

template grid_layout<float>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   const float *ptr,
                   const int rank);

template grid_layout<double>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   const double *ptr,
                   const int rank);

template grid_layout<std::complex<float>>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   const std::complex<float> *ptr,
                   const int rank);

template grid_layout<std::complex<double>>
get_scalapack_layout(int lld,
                   scalapack::matrix_dim m_dim,
                   scalapack::elem_grid_coord ij,
                   scalapack::matrix_dim subm_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   scalapack::rank_grid_coord rank_src,
                   const std::complex<double> *ptr,
                   const int rank);

template grid_layout<float>
get_scalapack_layout(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   float *ptr,
                   int rank);

template grid_layout<double>
get_scalapack_layout(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   double *ptr,
                   int rank);

template grid_layout<std::complex<float>>
get_scalapack_layout(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   std::complex<float> *ptr,
                   int rank);

template grid_layout<std::complex<double>>
get_scalapack_layout(scalapack::matrix_dim m_dim,
                   scalapack::block_dim b_dim,
                   scalapack::rank_decomposition r_grid,
                   scalapack::ordering rank_grid_ordering,
                   std::complex<double> *ptr,
                   int rank);

template grid_layout<float>
get_scalapack_layout(scalapack::data_layout &layout, float *ptr, int rank);

template grid_layout<double>
get_scalapack_layout(scalapack::data_layout &layout, double *ptr, int rank);

template grid_layout<std::complex<float>>
get_scalapack_layout(scalapack::data_layout &layout,
                   std::complex<float> *ptr,
                   int rank);

template grid_layout<std::complex<double>>
get_scalapack_layout(scalapack::data_layout &layout,
                   std::complex<double> *ptr,
                   int rank);
} // namespace costa
