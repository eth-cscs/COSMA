#include <cosma/blacs.hpp>

using namespace cosma;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int myrow, mycol, ctxt;
    char order = 'R';
    int p = 2;
    int q = 2;

    blacs::Cblacs_get(0, 0, &ctxt);
    blacs::Cblacs_gridinit(&ctxt, &order, p, q);
    blacs::Cblacs_gridinfo(ctxt, &p, &q, &myrow, &mycol);
    // blacs::Cblacs_pcoord(ctxt, rank, &myrow, &mycol);

    MPI_Comm comm = blacs::Cblacs2sys_handle(ctxt);
    MPI_Barrier(comm);

    blacs::Cblacs_gridexit(ctxt);
    MPI_Finalize();

    return 0;
}
