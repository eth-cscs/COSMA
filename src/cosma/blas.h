#ifndef COSMA_BLAS_H
#define COSMA_BLAS_H
#include <mpi.h>

namespace blas {

extern "C" {
/* Cblacs declarations */
void Cblacs_pinfo(int *, int *);
void Cblacs_get(int, int, int *);
void Cblacs_gridinit(int *, const char *, int, int);
void Cblacs_pcoord(int, int, int *, int *);
void Cblacs_gridexit(int);
void Cblacs_barrier(int, const char *);

int numroc(int *, int *, int *, int *, int *);

void pdgemr2d(int *m,
              int *n,
              double *a,
              int *ia,
              int *ja,
              int *desca,
              double *b,
              int *ib,
              int *jb,
              int *descb,
              int *ictxt);

void pdgemm(const char *trans_a,
            const char *transb,
            const int *m,
            const int *n,
            const int *k,
            const double *alpha,
            const double *a,
            const int *ia,
            const int *ja,
            const int *desca,
            const double *b,
            const int *ib,
            const int *jb,
            const int *descb,
            const double *beta,
            double *c,
            const int *ic,
            const int *jc,
            const int *descc);

void descinit(int *desc,
              int *m,
              int *n,
              int *bm,
              int *bn,
              int *rsrc,
              int *csrc,
              int *ctxt,
              int *lda,
              int *info);

void dgemm(char *,
           char *,
           int *,
           int *,
           int *,
           double *,
           double *,
           int *,
           double *,
           int *,
           double *,
           double *,
           int *);
void daxpy(int *, double *, double *, int *, double *, int *);

MPI_Comm Cblacs2sys_handle(int ictxt);
int Csys2blacs_handle(MPI_Comm mpi_comm);
void Cfree_blacs_system_handle(int i_sys_ctxt);
}
} // namespace blas
#endif
