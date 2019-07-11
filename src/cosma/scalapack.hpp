#ifdef COSMA_WITH_SCALAPACK
#pragma once
// from std
#include <cassert>
// from cosma
#include <cosma/blacs.hpp>
// from grid2grid
#include <scalapack_layout.hpp>

namespace cosma {
namespace scalapack {
struct block_sizes {
    int m = 0;
    int n = 0;
    int k = 0;

    block_sizes() = default;
    block_sizes(int m, int n, int k): m(m), n(n), k(k) {}
    block_sizes(const int* desca, const int* descb, const int* descc) {
        m = desca[4];
        k = desca[5];
        n = descb[5];

        // bk should be equal for A and B
        assert(desca[5] == descb[4]);
        // bm should be equal for A and C
        assert(desca[4] == descc[4]);
        // bn should be equal to B and C
        assert(descb[5] == descc[5]);
    }
};

struct global_matrix_sizes {
    int m = 0;
    int n = 0;
    int k = 0;

    global_matrix_sizes() = default;
    global_matrix_sizes(int m, int n, int k): m(m), n(n), k(k) {}
    global_matrix_sizes(const int* desca, const int* descb, const int* descc) {
        m = desca[2];
        k = desca[3];
        n = descb[3];

        // m_global should be equal for A and C
        assert(desca[2] == descc[2]);
        // k_global should be equal for A and B
        assert(desca[3] == descb[2]);
        // n_global should be equal to B and C
        assert(descb[3] == descc[3]);
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

int get_grid_context(const int* desca, const int* descb, const int* descc);
int get_comm_context(const int grid_context);
MPI_Comm get_communicator(const int comm_context);

int leading_dimension(const int* desc);
}}
#endif
