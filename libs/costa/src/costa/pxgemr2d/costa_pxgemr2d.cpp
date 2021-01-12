#include <cassert>
#include <mpi.h>

#include <costa/blacs.hpp>
#include <costa/pxgemr2d/pxgemr2d_params.hpp>
#include <costa/pxgemr2d/costa_pxgemr2d.hpp>
#include <costa/grid2grid/ranks_reordering.hpp>

#include <costa/grid2grid/transform.hpp>

#include <costa/grid2grid/profiler.hpp>

namespace costa {
template <typename T>
void pxgemr2d(
           const int m,
           const int n,
           const T *a,
           const int ia,
           const int ja,
           const int *desca,
           T *c,
           const int ic,
           const int jc,
           const int *descc,
           const int ctxt) {
    // clear the profiler
    // empty if compiled without profiler
    PC();

    // **********************************
    //           CORNER CASES
    // **********************************
    // edge cases, which are allowed by the standard
    if (m == 0 || n == 0) return;

    // **********************************
    //           MAIN CODE
    // **********************************
    // blas context
    int ctxt_a = scalapack::get_grid_context(desca);
    int ctxt_c = scalapack::get_grid_context(descc);

    // scalapack rank grid decomposition
    int procrows, proccols;
    int myrow, mycol;
    blacs::Cblacs_gridinfo(ctxt, &procrows, &proccols, &myrow, &mycol);

    // get MPI communicators
    MPI_Comm comm_a = scalapack::get_communicator(ctxt_a);
    MPI_Comm comm_c = scalapack::get_communicator(ctxt_c);
    /*
    MPI_Comm comm = scalapack::get_communicator(ctxt);
    // MPI_Comm comm = blacs::Cblacs2sys_handle(ctxt);
    // check if comm is at least the union of comm_a and comm_c
    assert(is_subcommunicator(comm, comm_a));
    assert(is_subcommunicator(comm, comm_c));
    */

    MPI_Comm comm = scalapack::comm_union(comm_a, comm_c);

    // communicator size and rank
    int rank, P;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    // block sizes
    scalapack::block_size b_dim_a(desca);
    scalapack::block_size b_dim_c(descc);

    // global matrix sizes
    scalapack::global_matrix_size mat_dim_a(desca);
    scalapack::global_matrix_size mat_dim_c(descc);

    // sumatrix size to multiply
    int a_subm = m;
    int a_subn = n;

    int c_subm = m;
    int c_subn = n;

    // rank sources (rank coordinates that own first row and column of a matrix)
    scalapack::rank_src rank_src_a(desca);
    scalapack::rank_src rank_src_c(descc);

    // leading dimensions
    int lld_a = scalapack::leading_dimension(desca);
    int lld_c = scalapack::leading_dimension(descc);

    // check whether rank grid is row-major or col-major
    auto ordering = scalapack::rank_ordering(ctxt, P);
    char grid_order =
        ordering == costa::scalapack::ordering::column_major ? 'C' : 'R';

#ifdef DEBUG
    if (rank == 0) {
        pxgemr2d_params<T> params(
                             // global dimensions
                             mat_dim_a.rows, mat_dim_a.cols,
                             mat_dim_c.rows, mat_dim_c.cols,
                             // block dimensions
                             b_dim_a.rows, b_dim_a.cols,
                             b_dim_c.rows, b_dim_c.cols,
                             // submatrix start
                             ia, ja,
                             ic, jc,
                             // problem size
                             m, n,
                             // leading dimensinons
                             lld_a, lld_c,
                             // processor grid
                             procrows, proccols,
                             // processor grid ordering
                             grid_order,
                             // ranks containing first rows
                             rank_src_a.row_src, rank_src_a.col_src,
                             rank_src_c.row_src, rank_src_c.col_src
                         );
        std::cout << params << std::endl;
    }
    MPI_Barrier(comm);
#endif

#ifdef DEBUG
    if (rank == 0) {
        std::cout << strategy << std::endl;
        std::cout << "============================================" << std::endl;
    }
    MPI_Barrier(comm);
#endif

    // get abstract layout descriptions for ScaLAPACK layout
    auto scalapack_layout_a = costa::get_scalapack_layout<T>(
        lld_a,
        {mat_dim_a.rows, mat_dim_a.cols},
        {ia, ja},
        {a_subm, a_subn},
        {b_dim_a.rows, b_dim_a.cols},
        {procrows, proccols},
        ordering,
        {rank_src_a.row_src, rank_src_a.col_src},
        a,
        rank);

    auto scalapack_layout_c = costa::get_scalapack_layout<T>(
        lld_c,
        {mat_dim_c.rows, mat_dim_c.cols},
        {ic, jc},
        {c_subm, c_subn},
        {b_dim_c.rows, b_dim_c.cols},
        {procrows, proccols},
        ordering,
        {rank_src_c.row_src, rank_src_c.col_src},
        c,
        rank);

    MPI_Barrier(comm);
    // transform A to C
    auto start = std::chrono::steady_clock::now();

    costa::transform<T>(scalapack_layout_a, scalapack_layout_c, comm);

    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();

    auto timing_no_relabeling =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    bool reordered = false;
    auto comm_vol = costa::communication_volume(scalapack_layout_a.grid, scalapack_layout_c.grid);
    std::vector<int> rank_permutation = costa::optimal_reordering(comm_vol, P, reordered);
    scalapack_layout_c.reorder_ranks(rank_permutation);

    MPI_Barrier(comm);
    // transform A to C
    start = std::chrono::steady_clock::now();

    costa::transform<T>(scalapack_layout_a, scalapack_layout_c, comm);

    MPI_Barrier(comm);
    end = std::chrono::steady_clock::now();

    auto timing_with_relabeling =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    auto new_comm_vol = costa::communication_volume(scalapack_layout_a.grid, scalapack_layout_c.grid);

    auto comm_vol_total = comm_vol.total_volume();
    auto new_comm_vol_total = new_comm_vol.total_volume();


    std::cout << "Time no relabeling [ms] = " << timing_no_relabeling << std::endl;
    std::cout << "Time with relabeling [ms] = " << timing_with_relabeling << std::endl;

    auto diff = (long long) comm_vol_total - (long long) new_comm_vol_total;
    if (comm_vol_total > 0) {
        std::cout << "Comm volume reduction [%] = " << 100.0 * diff / comm_vol_total << std::endl;
    } else {
        std::cout << "Initial comm vol = 0, nothing to improve." << std::endl;
    }

    // print the profiling data
    if (rank == 0) {
        PP();
    }
}

// explicit instantiation for pxtran
template void pxgemr2d<double>(
                            const int m,
                            const int n,
                            const double *a,
                            const int ia,
                            const int ja,
                            const int *desca,
                            double *c,
                            const int ic,
                            const int jc,
                            const int *descc,
                            const int ctxt);

template void pxgemr2d<float>(
                           const int m,
                           const int n,
                           const float *a,
                           const int ia,
                           const int ja,
                           const int *desca,
                           float *c,
                           const int ic,
                           const int jc,
                           const int *descc,
                           const int ctxt);

template void pxgemr2d<zdouble_t>(
                               const int m,
                               const int n,
                               const zdouble_t *a,
                               const int ia,
                               const int ja,
                               const int *desca,
                               zdouble_t *c,
                               const int ic,
                               const int jc,
                               const int *descc,
                               const int ctxt);

template void pxgemr2d<zfloat_t>(
                              const int m,
                              const int n,
                              const zfloat_t *a,
                              const int ia,
                              const int ja,
                              const int *desca,
                              zfloat_t *c,
                              const int ic,
                              const int jc,
                              const int *descc,
                              const int ctxt);
} // namespace costa
