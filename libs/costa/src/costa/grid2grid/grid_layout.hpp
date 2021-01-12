#pragma once
#include <costa/grid2grid/block.hpp>
#include <costa/grid2grid/grid2D.hpp>
#include <costa/grid2grid/mpi_type_wrapper.hpp>

namespace costa {
template <typename T>
class grid_layout {
  public:
    grid_layout() = default;

    grid_layout(assigned_grid2D &&g, local_blocks<T> &&b)
        : grid(std::forward<assigned_grid2D>(g))
        , blocks(std::forward<local_blocks<T>>(b)) {}

    int num_ranks() const { return grid.num_ranks(); }

    void transpose_or_conjugate(char flag) {
        flag = std::toupper(flag);
        assert(flag == 'N' || flag == 'T' || flag == 'C');
        if (flag == 'T' || flag == 'C') {
            grid.transpose();
            blocks.transpose_or_conjugate(flag);
        }
    }

    void reorder_ranks(std::vector<int>& reordering) {
        grid.reorder_ranks(reordering);
    }

    int reordered_rank(int rank) const {
        return grid.reordered_rank(rank);
    }

    bool ranks_reordered() {
        return grid.ranks_reordered();
    }

    // returns the number of rows and column in the matrix
    // that this grid represents, and not the number of blocks
    // in a row or column of the grid
    int num_cols() const noexcept { return grid.num_cols(); }
    int num_rows() const noexcept { return grid.num_rows(); }

    // returns the number of blocks in a row/column of the grid
    int num_blocks_col() const noexcept { return grid.num_blocks_col(); }
    int num_blocks_row() const noexcept { return grid.num_blocks_row(); }

    // scales all local blocks by beta
    void scale_by(const T beta) {
        if (beta == T{1}) return;
        // iterate over all local blocks
        // local_blocks contains only blocks within the specified submatrix
        for (unsigned i = 0u; i < blocks.num_blocks(); ++i) {
            auto& block = blocks.get_block(i);
            block.scale_by(beta);
        }
    }

    // initializes the matrix based on the given lambda function
    // f that maps global coordinates (gi, gj) to the value of that element
    // f(gi, gj) := value of element with global coordinates (gi, gj)
    template <typename Function>
    void initialize(Function f) {
        using elem_type = T;

        for (size_t i = 0; i < blocks.num_blocks(); ++i) {
            block<T> b = blocks.get_block(i);

            // iterate over local coordinates
            for (int li = 0; li < b.n_rows(); ++li) {
                for (int lj = 0; lj < b.n_cols(); ++lj) {
                    int gi, gj;
                    // local -> global coordinates
                    std::tie(gi, gj) = b.local_to_global(li, lj);
                    // check if global coordinates within global matrix dims
                    assert(gi >= 0 && gj >= 0);
                    assert(gi < num_rows() && gj < num_cols());
                    // initialize local elemenent by f(global coordinates)
                    b.local_element(li, lj) = (T) f(gi, gj);
                    /*
                    std::cout << "mat(" << gi << ", " << gj << ")"
                              << " = " << (T) f(gi, gj)
                              << std::endl;
                    */
                }
            }
        }
    }

    // checks whether the matrix elements correspond to the value of function f
    // i.e. it checks if:
    // global element (i, j) is equal to f(i, j) for all i, j
    // taking into account the tolerance
    template <typename Function>
    bool validate(Function f, double tolerance = 1e-12) {
        using elem_type = T;
        bool ok = true;

        for (size_t i = 0; i < blocks.num_blocks(); ++i) {
            block<T> b = blocks.get_block(i);

            // iterate over local coordinates
            for (int li = 0; li < b.n_rows(); ++li) {
                for (int lj = 0; lj < b.n_cols(); ++lj) {
                    int gi, gj;
                    // local -> global coordinates
                    std::tie(gi, gj) = b.local_to_global(li, lj);
                    // check if global coordinates within global matrix dims
                    assert(gi >= 0 && gj >= 0);
                    assert(gi < num_rows() && gj < num_cols());
                    // initialize local elemenent by f(global coordinates)
                    auto diff = std::abs(b.local_element(li, lj) - (T)f(gi, gj));
                    if (diff > tolerance) { 
                        std::cout << "[ERROR] mat(" << gi << ", " << gj << ")"
                                  <<  " = " << b.local_element(li, lj)
                                  << " instead of " << (T) f(gi, gj)
                                  << std::endl;
                        ok = false;
                        // return false;
                    }
                }
            }
        }
        return ok;
    }

    assigned_grid2D grid;
    local_blocks<T> blocks;
};

template <typename T>
using layout_ref = std::reference_wrapper<grid_layout<T>>;

} // namespace costa
