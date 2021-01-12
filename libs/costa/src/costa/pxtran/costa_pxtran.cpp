#include <cassert>
#include <mpi.h>

#include <costa/blacs.hpp>
#include <costa/pxtran/costa_pxtran.hpp>

#include <costa/grid2grid/ranks_reordering.hpp>
#include <costa/grid2grid/transformer.hpp>
#include <costa/grid2grid/cantor_mapping.hpp>

#include <costa/grid2grid/profiler.hpp>

namespace costa {
template <typename T>
void pxtran(
           const int m,
           const int n,
           const T alpha,
           const T *a,
           const int ia,
           const int ja,
           const int *desca,
           const T beta,
           T *c, // result
           const int ic,
           const int jc,
           const int *descc) {
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
    int ctxt = scalapack::get_grid_context(desca, descc);

    // scalapack rank grid decomposition
    int procrows, proccols;
    int myrow, mycol;
    blacs::Cblacs_gridinfo(ctxt, &procrows, &proccols, &myrow, &mycol);

    // get MPI communicator
    MPI_Comm comm = scalapack::get_communicator(ctxt);

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
    int a_subm = n;
    int a_subn = m;

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
        pxtran_params<T> params(
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
                             // alpha, beta
                             alpha, beta,
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

/*
#ifdef DEBUG
    auto values = [=](int i, int j) -> T {
        int el = cantor_pairing(i, j);
        return static_cast<T>(el);
    };

    scalapack_layout_a.initialize(values);
#endif
*/

    costa::transform<T>(scalapack_layout_a, scalapack_layout_c, 'T', alpha, beta, comm);

/*
#ifdef DEBUG
    auto transposed_values = [=](int i, int j) -> T {
        int el = cantor_pairing(j, i);
        return static_cast<T>(el);
    };

    scalapack_layout_c.validate(transposed_values, 1e-6);
#endif
*/

    // print the profiling data
    if (rank == 0) {
        PP();
    }
}

// explicit instantiation for pxtran
template void pxtran<double>(
                            const int m,
                            const int n,
                            const double alpha,
                            const double *a,
                            const int ia,
                            const int ja,
                            const int *desca,
                            const double beta,
                            double *c,
                            const int ic,
                            const int jc,
                            const int *descc);

template void pxtran<float>(
                           const int m,
                           const int n,
                           const float alpha,
                           const float *a,
                           const int ia,
                           const int ja,
                           const int *desca,
                           const float beta,
                           float *c,
                           const int ic,
                           const int jc,
                           const int *descc);

template void pxtran<zdouble_t>(
                               const int m,
                               const int n,
                               const zdouble_t alpha,
                               const zdouble_t *a,
                               const int ia,
                               const int ja,
                               const int *desca,
                               const zdouble_t beta,
                               zdouble_t *c,
                               const int ic,
                               const int jc,
                               const int *descc);

template void pxtran<zfloat_t>(
                              const int m,
                              const int n,
                              const zfloat_t alpha,
                              const zfloat_t *a,
                              const int ia,
                              const int ja,
                              const int *desca,
                              const zfloat_t beta,
                              zfloat_t *c,
                              const int ic,
                              const int jc,
                              const int *descc);
} // namespace costa
