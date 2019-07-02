#pragma once
#include <complex>
#include <mpi.h>

#ifdef COSMA_WITH_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace cosma {
// Cblacs initialization
void Cblacs_pinfo(int* mypnum, int* nprocs);
void Cblacs_gridinit(int* ictxt, char* order, int nprow, int npcol);
void Cblacs_set(int ictxt, int what, int* val);
void Cblacs_get(int ictxt, int what, int* val);
void Cblacs_setup(int* mypnum, int* nprocs);
void Cblacs_gridmap(int* ictxt, int* usermap, int ldup, int nprow, int npcol);

// Cblacs finalization and aborting
void Cblacs_gridexit(int ictxt);
void Cblacs_freebuff(int ictxt, int wait);
void Cblacs_exit(int NotDone);
void Cblacs_abort(int ictxt, int errno);

// processor grid information
void Cblacs_gridinfo(int ictxt, int* nprow, int* npcol, int* myrow, int* mycol);
int Cblacs_pnum(int ictxt, int prow, int pcol);
void Cblacs_pcoord(int ictxt, int nodenum, int* prow, int* pcol);

// switching between MPI communicator and Cblacs context
MPI_Comm Cblacs2sys_handle(int ictxt);
void Cfree_blacs_system_handle(int i_sys_ctxt);
int Csys2blacs_handle(MPI_Comm mpi_comm);

// barrier
void Cblacs_barrier(int ictxt, char* scope);

void dgemm(const int M,
           const int N,
           const int K,
           const double alpha,
           const double *A,
           const int lda,
           const double *B,
           const int ldb,
           const double beta,
           double *C,
           const int ldc);

void dgemm(const int M,
           const int N,
           const int K,
           const std::complex<double> alpha,
           const std::complex<double> *A,
           const int lda,
           const std::complex<double> *B,
           const int ldb,
           const std::complex<double> beta,
           std::complex<double> *C,
           const int ldc);

void dgemm(const int M,
           const int N,
           const int K,
           const float alpha,
           const float *A,
           const int lda,
           const float *B,
           const int ldb,
           const float beta,
           float *C,
           const int ldc);

void dgemm(const int M,
           const int N,
           const int K,
           const std::complex<float> alpha,
           const std::complex<float> *A,
           const int lda,
           const std::complex<float> *B,
           const int ldb,
           const std::complex<float> beta,
           std::complex<float> *C,
           const int ldc);
} // namespace cosma
