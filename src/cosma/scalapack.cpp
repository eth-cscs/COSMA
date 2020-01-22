#include <cosma/scalapack.hpp>

grid2grid::scalapack::ordering cosma::scalapack::rank_ordering(int ctxt, int P) {
    // check whether rank grid is row-major or col-major
    auto ordering = grid2grid::scalapack::ordering::column_major;
    if (P > 1) {
        int prow, pcol;
        // check the coordinates of rank 1 to see
        // if the rank grid is row-major or col-major
        blacs::Cblacs_pcoord(ctxt, 1, &prow, &pcol);
        if (prow == 0 && pcol == 1) {
            ordering = grid2grid::scalapack::ordering::row_major;
        }
    }
    return ordering;
}

int cosma::scalapack::get_grid_context(const int* desca, const int* descb, const int* descc) {
    int ctxt = desca[1];
    // all matrices should belong to the same context
    assert(desca[1] == descb[1]);
    assert(descb[1] == descc[1]);
    return ctxt;
}

int cosma::scalapack::leading_dimension(const int* desc) {
    return desc[8];
}

// queries the grid blacs context to get the communication blacs context
int cosma::scalapack::get_comm_context(const int grid_context) {
    int comm_context;
    blacs::Cblacs_get(grid_context, 10, &comm_context);
    return comm_context;
}

// gets MPI_Comm from the grid blacs context
MPI_Comm cosma::scalapack::get_communicator(const int grid_context) {
    int comm_context = get_comm_context(grid_context);
    MPI_Comm comm = blacs::Cblacs2sys_handle(comm_context);
    return comm;
}

// computes the number of rows or columns that the specified rank owns
int cosma::scalapack::numroc(int n, int nb, int proc_coord, int proc_src, int n_procs) {
    // Arguments:
    /*
      - n: global matrix dimension (rows or columns)
      - nb: corresponding dimension of a block
      - proc_coord: coordinate of the process for which we are querying
      - proc_src: process src
      - n_procs: total number of processes along this dimension
     */
    // number of whole blocks along this dimension
    int n_blocks = n / nb;

    // the offset of given process to the source process
    // make sure it stays positive
    int proc_offset = (n_procs + proc_coord - proc_src) % n_procs;

    // number of blocks per process (at least)
    // Can also be zero.
    int n_blocks_per_process = n_blocks/n_procs;
    // Number of rows or columns that each process has (at least).
    // Can also be zero.
    int n_rows_or_cols_per_process = n_blocks_per_process * nb;

    // each rank owns at least this base
    int n_rows_or_columns_total = n_rows_or_cols_per_process;

    // if there is a remainder, then the current 
    // process might own some additional blocks
    int remainder = n_blocks % n_procs;

    // possible additional "whole" blocks that
    // the current rank owns
    n_rows_or_columns_total += proc_offset < remainder ? nb : 0;
    // possible additional "partial" blocks that 
    // the current ranks owns
    n_rows_or_columns_total += proc_offset == remainder ? n % nb : 0;

    return n_rows_or_columns_total;
}

// minimum lld: used mostly for correctness checking of pxgemm parameters
int cosma::scalapack::min_leading_dimension(int n, int nb, int rank_grid_dim) {
    // Arguments:
    /*
      - n: global matrix dimension (rows or columns)
      - nb: corresponding dimension of a block
      - rank_grid_dim: total number of processes along this dimension
     */
    // number of blocks along this dimension
    int n_blocks = n / nb;

    // number of blocks per process (at least)
    // Can also be zero.
    int n_blocks_per_process = n_blocks/rank_grid_dim;
    // Number of rows or columns that each process has (at least).
    // Can also be zero.
    // each rank owns at least this many rows
    int min_n_rows_or_cols_per_process = n_blocks_per_process * nb;

    return min_n_rows_or_cols_per_process;
}

// maximum lld: used mostly for correctness checking of pxgemm parameters
int cosma::scalapack::max_leading_dimension(int n, int nb, int rank_grid_dim) {
    // Arguments:
    /*
      - n: global matrix dimension (rows or columns)
      - nb: corresponding dimension of a block
      - rank_grid_dim: total number of processes along this dimension
     */
    return nb + min_leading_dimension(n, nb, rank_grid_dim);
}

int cosma::scalapack::local_buffer_size(const int* desc) {
    int lld = leading_dimension(desc);

    int n_cols = desc[3]; // global matrix size (columns)
    int nb_cols = desc[5]; // block size (columns)
    int src_proc = desc[7]; // processor src (columns)

    int ctxt = desc[1];

    int nprow, npcol, myrow, mycol;
    blacs::Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

    int P = nprow * npcol;

    int n_local_cols = numroc(n_cols, nb_cols, mycol, src_proc, npcol);

    return lld * n_local_cols;
}
