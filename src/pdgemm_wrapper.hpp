#include <mpi.h>
#include "blas.h"

namespace cosma {
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

    int rsrc_a = desca[6];
    int csrc_a = desca[7];
    int lld_a = desca[8];

    int rsrc_b = descb[6];
    int csrc_b = descb[7];
    int lld_b = descb[8];

    int rsrc_c = descc[6];
    int csrc_c = descc[7];
    int lld_c = descc[8];

    // Release resources
    Cblacs_gridexit(ctxt);
    Cfree_blacs_system_handle(comm);
}
}
