#ifndef _BLAS_H_
#define _BLAS_H_

extern "C" {
    void dpotrf_(char*, int*, double*, int*, int*);
    void dpptrf_(char*, int*, double*, int*);
    void dgemm_(char*, char*, int*,int*,int*, double*, double*,int*, double*,int*, double*, double*,int*);
    void daxpy_(int*, double*, double*, int*, double*, int*);
}

#endif 
