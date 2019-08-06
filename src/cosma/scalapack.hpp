#pragma once

#include <cosma/blacs.hpp>

#include <grid2grid/scalapack_layout.hpp>

#include <cassert>

namespace cosma {
namespace scalapack {
struct block_size {
    int rows = 0;
    int cols = 0;

    block_size() = default;
    block_size(int rows, int cols): rows(rows), cols(cols) {}
    block_size(const int* desc) {
        rows = desc[4];
        cols = desc[5];
    }
};

struct global_matrix_size {
    int rows = 0;
    int cols = 0;

    global_matrix_size() = default;
    global_matrix_size(int rows, int cols): rows(rows), cols(cols) {}
    global_matrix_size(const int* desc) {
        rows = desc[2];
        cols = desc[3];
    }
};

struct rank_src {
    int row_src = 0;
    int col_src = 0;

    rank_src() = default;
    rank_src(int rsrc, int csrc): row_src(rsrc), col_src(csrc) {}
    rank_src(const int* desc) {
        row_src = desc[6];
        col_src = desc[7];
    }
};

grid2grid::scalapack::ordering rank_ordering(int ctxt, int P);

// gets the grid context from descriptors of A, B and C and compares
// if all three matrices belong to the same context
int get_grid_context(const int* desca, const int* descb, const int* descc);
// gets the communication blacs context from the grid blacs context
int get_comm_context(const int grid_context);
// gets the MPI communicator from the grid blacs context
MPI_Comm get_communicator(const int grid_context);

int leading_dimension(const int* desc);
}}
