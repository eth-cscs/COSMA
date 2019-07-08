#pragma once
#include <mpi.h>

#ifdef COSMA_WITH_MKL
#include <mkl_blacs.h>
#else
#include <cblacs.h>
#endif

namespace cosma {
extern "C" {
    void descinit(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);
    int numroc(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

    // void pdgemm(const char trans_a, const char trans_b, const int m, const int n, const int k,
    //            const double alpha, const double* a, const int ia, const int ja, const int* desca,
    //            const double* b, const int ib, const int jb, const int* descb, const double beta,
    //            double* c, const int ic, const int jc, const int* descc) {

    // Initialization
    void Cblacs_pinfo(int* mypnum, int* nprocs);
    void Cblacs_setup(int* mypnum, int* nprocs);
    void Cblacs_set(int ictxt, int what, int* val);
    void Cblacs_get(int ictxt, int what, int* val);
    void Cblacs_gridinit(int* ictxt, char* order, int nprow, int npcol);
    void Cblacs_gridmap(int* ictxt, int* usermap, int ldup, int nprow, int npcol);

    // Finalization
    void Cblacs_freebuff(int ictxt, int wait);
    void Cblacs_gridexit(int ictxt);
    void Cblacs_exit(int NotDone);

    // Abort
    void Cblacs_abort(int ictxt, int errno);

    // Information
    void Cblacs_gridinfo(int ictxt, int* nprow, int* npcol, int* myrow, int* mycol);
    int Cblacs_pnum(int ictxt, int prow, int pcol);
    void Cblacs_pcoord(int ictxt, int nodenum, int* prow, int* pcol);

    // Barrier
    void Cblacs_barrier(int ictxt, char* scope);

    // MPI communicator <-> Blacs context
    MPI_Comm Cblacs2sys_handle(int ictxt);
    int Csys2blacs_handle(MPI_Comm mpi_comm);
    void Cfree_blacs_system_handle(int i_sys_ctxt);
}
}

