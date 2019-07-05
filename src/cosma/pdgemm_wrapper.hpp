#include <mpi.h>
#include <cosma/blacs.hpp>
#include <cosma/multiply.hpp>
#include <transform.hpp>

namespace cosma {
template<typename T>
// alpha ignored at the moment
void pgemm(const char trans_a, const char trans_b, const int m, const int n, const int k,
           const double alpha, const double* a, const int ia, const int ja, const int* desca,
           const double* b, const int ib, const int jb, const int* descb, const double beta,
           double* c, const int ic, const int jc, const int* descc) {
    int iZERO = 0;
    int ctxt, myrow, mycol;
    int rank, P;

    ctxt = desca[1];

    Cblacs_pinfo(&rank, &P);
    // Cblacs_gridinit(&ctxt, "Row-major", procrows, proccols);
    int procrows, proccols;
    Cblacs_gridinfo(ctxt, &procrows, &proccols, &myrow, &mycol);
    // Cblacs_pcoord(ctxt, myid, &myrow, &mycol);
    MPI_Comm comm = Cblacs2sys_handle(ctxt);
    std::cout << "MPI comm = " << comm << std::endl;

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

    bool trans_a_flag = trans_a == 'N';
    bool trans_b_flag = trans_b == 'N';
    bool trans_c_flag = 'N';

    // check whether rank grid is row-major or col-major
    auto ordering = grid2grid::scalapack::ordering::column_major;

    if (P > 1) {
        int prow, pcol;
        // check the coordinates of rank 1 to see
        // if the rank grid is row-major or col-major
        Cblacs_pcoord(ctxt, 1, &prow, &pcol);
        if (prow == 0 && pcol == 1) {
            ordering = grid2grid::scalapack::ordering::row_major;
        }
    }

    // find an optimal strategy for this problem
    Strategy strategy(m, n, k, P);
    std::cout << strategy << std::endl;

    // create COSMA matrices
    CosmaMatrix<T> A('A', strategy, rank);
    CosmaMatrix<T> B('B', strategy, rank);
    CosmaMatrix<T> C('C', strategy, rank);

    // get abstract layout descriptions for COSMA layout
    std::cout << "Getting COSMA grid for matrix A." << std::endl;
    auto cosma_layout_a = A.get_grid_layout();
    std::cout << "Getting COSMA grid for matrix B." << std::endl;
    auto cosma_layout_b = B.get_grid_layout();
    std::cout << "Getting COSMA grid for matrix C." << std::endl;
    auto cosma_layout_c = C.get_grid_layout();

    std::cout << "Getting scalapack grid for matrix A." << std::endl;
    // get abstract layout descriptions for ScaLAPACK layout
    auto scalapack_layout_a = grid2grid::get_scalapack_grid<T>(
        lld_a,
        {m_global, k_global},
        {ia, ja},
        {m, k},
        {bm, bk},
        {procrows, proccols},
        ordering,
        trans_a_flag,
        {rsrc_a, csrc_a},
        const_cast<T*>(a), rank
    );

    std::cout << "Getting scalapack grid for matrix B." << std::endl;
    auto scalapack_layout_b = grid2grid::get_scalapack_grid<T>(
        lld_b,
        {k_global, n_global},
        {ib, jb},
        {k, n},
        {bk, bn},
        {procrows, proccols},
        ordering,
        trans_b_flag,
        {rsrc_b, csrc_b},
        const_cast<T*>(b), rank
    );

    std::cout << "Getting scalapack grid for matrix C." << std::endl;
    auto scalapack_layout_c = grid2grid::get_scalapack_grid<T>(
        lld_c,
        {m_global, n_global},
        {ic, jc},
        {m, n},
        {bm, bn},
        {procrows, proccols},
        ordering,
        trans_c_flag,
        {rsrc_c, csrc_c},
        c, rank
    );

    std::cout << "Matrices ready for transformation." << std::endl;

    // transform A and B from scalapack to cosma layout
    grid2grid::transform(scalapack_layout_a, cosma_layout_a, comm);
    std::cout << "A: scalapack->cosma finished" << std::endl;
    grid2grid::transform(scalapack_layout_b, cosma_layout_b, comm);
    std::cout << "B: scalapack->cosma finished" << std::endl;
    // grid2grid::transform(scalapack_layout_c, cosma_layout_c, comm);

    std::cout << "Starting COSMA algorithm." << std::endl;
    auto ctx = cosma::make_context();
    // perform cosma multiplication
    multiply(ctx, A, B, C, strategy, comm, beta);

    std::cout << "Finished COSMA algorithm." << std::endl;

    // transform the result from cosma back to scalapack
    grid2grid::transform(cosma_layout_c, scalapack_layout_c, comm);
    std::cout << "C: cosma->scalapack finished" << std::endl;

    // Release resources
    // Cblacs_gridexit(ctxt);
    // Cfree_blacs_system_handle(comm);
}
}
