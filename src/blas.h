#ifndef BLAS_H
#define BLAS_H

extern "C" {
    void dpotrf_(char*, int*, double*, int*, int*);
    void dpptrf_(char*, int*, double*, int*);
    void dgemm_(char*, char*, int*,int*,int*, double*, double*,int*, double*,int*, double*, double*,int*);
    void daxpy_(int*, double*, double*, int*, double*, int*);

    void pdgemm_(char*, char*, int* m, int* n, int* k, double* alpha,
        double * A, int * IA, int * JA, int * DESCA,
        double * B, int * IB, int * JB, int * DESCB,
        double * beta,
        double * C, int * IC, int * JC, int * DESCC);
}

#endif 
