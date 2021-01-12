#pragma once
#include <mpi.h>

namespace costa {
namespace blacs {
extern "C" {
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
}}

