#include <mpi.h>
#include <cosma/blas.hpp>

namespace cosma {
template<typename T>
// alpha ignored at the moment
void pdgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
           const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
           const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
           double* c, const int* ic, const int* jc, const int* descc) {
    int iZERO = 0;
    int ctxt, myrow, mycol;
    int rank, P;

    Cblacs_pinfo(&rank, &P);
    // Cblacs_gridinit(&ctxt, "Row-major", procrows, proccols);
    Cblacs_gridinfo(&ctxt, &procrows, &proccols, &myrow, &mycol);
    // Cblacs_pcoord(ctxt, myid, &myrow, &mycol);
    MPI_Comm comm = Cblacs2sys_handle(ctxt);

    // Number of rows and cols owned by the current process
    int nrows1 = numroc_(&m, &bm1, &myrow, &iZERO, &procrows);
    int ncols1 = numroc_(&n, &bn1, &mycol, &iZERO, &proccols);

    int nrows2 = numroc_(&m, &bm2, &myrow, &iZERO, &procrows);
    int ncols2 = numroc_(&n, &bn2, &mycol, &iZERO, &proccols);

    int ctxt = desca[1];
    int bm = desca[4];
    int bk = descb[4];
    int bn = descc[4];

    int m_global = desca[2];
    int k_global = desca[3];
    int n_global = descb[3];

    int rsrc_a = desca[6];
    int csrc_a = desca[7];
    int lld_a = desca[8];

    int rsrc_b = descb[6];
    int csrc_b = descb[7];
    int lld_b = descb[8];

    int rsrc_c = descc[6];
    int csrc_c = descc[7];
    int lld_c = descc[8];

    // find an optimal strategy for this problem
    Strategy strategy(m, n, k, P);

    // create COSMA matrices
    CosmaMatrix<double> A('A', strategy, rank);
    CosmaMatrix<double> B('B', strategy, rank);
    CosmaMatrix<double> C('C', strategy, rank);

    // get abstract layout descriptions for COSMA layout
    grid2grid::grid_layout<T> cosma_layout_a = A.get_grid_layout();
    grid2grid::grid_layout<T> cosma_layout_b = B.get_grid_layout();
    grid2grid::grid_layout<T> cosma_layout_c = C.get_grid_layout();

    // get abstract layout descriptions for ScaLAPACK layout
    grid2grid::grid_layout<T> scalapack_layout_a = grid2grid::get_scalapack_grid(
        lld_a,
        {m_global, k_global},
        {*ia, *ja},
        {*m, *k},
        {bm, bk},
        {procrows, proccols},
        scalapack::ordering::col_major,
        transa,
        {rsrc_a, csrc_a},
        a, rank
    );

    grid2grid::grid_layout<T> scalapack_layout_b = grid2grid::get_scalapack_grid(
        lld_b,
        {k_global, n_global},
        {*ib, *jb},
        {*k, *n},
        {bk, bn},
        {procrows, proccols},
        scalapack::ordering::col_major,
        transb,
        {rsrc_b, csrc_b},
        b, rank
    );

    grid2grid::grid_layout<T> scalapack_layout_c = grid2grid::get_scalapack_grid(
        lld_c,
        {m_global, n_global},
        {*ic, *jc},
        {*m, *n},
        {bm, bn},
        {procrows, proccols},
        scalapack::ordering::col_major,
        transc,
        {rsrc_c, csrc_c},
        c, rank
    );

    // transform A and B from scalapack to cosma layout
    grid2grid::transform(scalapack_layout_a, cosma_layout_a, comm);
    grid2grid::transform(scalapack_layout_b, cosma_layout_b, comm);
    // grid2grid::transform(scalapack_layout_c, cosma_layout_c, comm);

    auto ctx = cosma::make_context();
    // perform cosma multiplication
    multiply(ctx, A, B, C, strategy, comm, beta);

    // transform the result from cosma back to scalapack
    grid2grid::transform(cosma_layout_c, scalapack_layout_c, comm);

    // Release resources
    Cblacs_gridexit(ctxt);
    Cfree_blacs_system_handle(comm);
}
}
