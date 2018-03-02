#ifndef BLAS_H
#define BLAS_H

extern "C" {
  void dpotrf_( char*, int*, double*, int*, int* );
  void dpptrf_( char*, int*, double*, int* );
  void dgemm_( char*, char*, int*,int*,int*, double*, double*,int*, double*,int*, double*, double*,int*);
  void dtrsm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);
  void dsyrk_( char*, char*, int*, int*, double*, double*, int*, double*, double*, int* );
  void daxpy_( int*, double*, double*, int*, double*, int* );
}

#endif 
