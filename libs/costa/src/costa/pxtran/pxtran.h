#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// scalapack api
void pstran(const int *m , const int *n , 
            float *alpha , const float *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const float *beta , float *c , 
            const int *ic , const int *jc ,
            const int *descc );

void pdtran(const int *m , const int *n , 
            double *alpha , const double *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const double *beta , double *c , 
            const int *ic , const int *jc ,
            const int *descc );

void pctranu(const int *m , const int *n , 
             float *alpha , const float *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const float *beta , float *c , 
             const int *ic , const int *jc ,
             const int *descc );

void pztranu(const int *m , const int *n , 
             double *alpha , const double *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const double *beta , double *c , 
             const int *ic , const int *jc ,
             const int *descc );

// *********************************************************************************
// Same as previously, but with added underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void pstran_(const int *m , const int *n , 
            float *alpha , const float *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const float *beta , float *c , 
            const int *ic , const int *jc ,
            const int *descc );

void pdtran_(const int *m , const int *n , 
            double *alpha , const double *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const double *beta , double *c , 
            const int *ic , const int *jc ,
            const int *descc );

void pctranu_(const int *m , const int *n , 
              float *alpha , const float *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const float *beta , float *c , 
              const int *ic , const int *jc ,
              const int *descc );

void pztranu_(const int *m , const int *n , 
              double *alpha , const double *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const double *beta , double *c , 
              const int *ic , const int *jc ,
              const int *descc );

// *********************************************************************************
// Same as previously, but with added double underscores at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void pstran__(const int *m , const int *n , 
              float *alpha , const float *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const float *beta , float *c , 
              const int *ic , const int *jc ,
              const int *descc );

void pdtran__(const int *m , const int *n , 
              double *alpha , const double *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const double *beta , double *c , 
              const int *ic , const int *jc ,
              const int *descc );

void pctranu__(const int *m , const int *n , 
               float *alpha , const float *a , 
               const int *ia , const int *ja , 
               const int *desca , 
               const float *beta , float *c , 
               const int *ic , const int *jc ,
               const int *descc );

void pztranu__(const int *m , const int *n , 
               double *alpha , const double *a , 
               const int *ia , const int *ja , 
               const int *desca , 
               const double *beta , double *c , 
               const int *ic , const int *jc ,
               const int *descc );

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void PSTRAN(const int *m , const int *n , 
            float *alpha , const float *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const float *beta , float *c , 
            const int *ic , const int *jc ,
            const int *descc );

void PDTRAN(const int *m , const int *n , 
            double *alpha , const double *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const double *beta , double *c , 
            const int *ic , const int *jc ,
            const int *descc );

void PCTRANU(const int *m , const int *n , 
             float *alpha , const float *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const float *beta , float *c , 
             const int *ic , const int *jc ,
             const int *descc );

void PZTRANU(const int *m , const int *n , 
             double *alpha , const double *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const double *beta , double *c , 
             const int *ic , const int *jc ,
             const int *descc );
#ifdef __cplusplus
}
#endif
