#include <costa/pxtran/costa_pxtran.hpp>

extern "C" {
#include <costa/pxtran/pxtran.h>

// Reimplement ScaLAPACK signatures functions
void pdtran(const int *m , const int *n , 
            double *alpha , const double *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const double *beta , double *c , 
            const int *ic , const int *jc ,
            const int *descc ) {
    costa::pxtran<double>(
                  *m,
                  *n,
                  *alpha,
                  a,
                  *ia,
                  *ja,
                  desca,
                  *beta,
                  c,
                  *ic,
                  *jc,
                  descc);
}

void pstran(const int *m , const int *n , 
            float *alpha , const float *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const float *beta , float *c , 
            const int *ic , const int *jc ,
            const int *descc ) {
    costa::pxtran<float>(
                  *m,
                  *n,
                  *alpha,
                  a,
                  *ia,
                  *ja,
                  desca,
                  *beta,
                  c,
                  *ic,
                  *jc,
                  descc);
}

void pctranu(const int *m , const int *n , 
             float *alpha , const float *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const float *beta , float *c , 
             const int *ic , const int *jc ,
             const int *descc ) {
    costa::pxtran<std::complex<float>>(
                  *m,
                  *n,
                  reinterpret_cast<const std::complex<float>&>(*alpha),
                  reinterpret_cast<const std::complex<float>*>(a),
                  *ia,
                  *ja,
                  desca,
                  reinterpret_cast<const std::complex<float>&>(*beta),
                  reinterpret_cast<std::complex<float>*>(c),
                  *ic,
                  *jc,
                  descc);
}

void pztranu(const int *m , const int *n , 
             double *alpha , const double *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const double *beta , double *c , 
             const int *ic , const int *jc ,
             const int *descc) {
    costa::pxtran<std::complex<double>>(
                  *m,
                  *n,
                  reinterpret_cast<const std::complex<double>&>(*alpha),
                  reinterpret_cast<const std::complex<double>*>(a),
                  *ia,
                  *ja,
                  desca,
                  reinterpret_cast<const std::complex<double>&>(*beta),
                  reinterpret_cast<std::complex<double>*>(c),
                  *ic,
                  *jc,
                  descc);
}

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
            const int *descc ) {
    pstran(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pdtran_(const int *m , const int *n , 
            double *alpha , const double *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const double *beta , double *c , 
            const int *ic , const int *jc ,
            const int *descc ) {
    pdtran(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pctranu_(const int *m , const int *n , 
              float *alpha , const float *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const float *beta , float *c , 
              const int *ic , const int *jc ,
              const int *descc ) {
    pctranu(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pztranu_(const int *m , const int *n , 
              double *alpha , const double *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const double *beta , double *c , 
              const int *ic , const int *jc ,
              const int *descc ) {
    pztranu(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

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
              const int *descc ) {
    pstran(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pdtran__(const int *m , const int *n , 
              double *alpha , const double *a , 
              const int *ia , const int *ja , 
              const int *desca , 
              const double *beta , double *c , 
              const int *ic , const int *jc ,
              const int *descc ) {
    pdtran(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pctranu__(const int *m , const int *n , 
               float *alpha , const float *a , 
               const int *ia , const int *ja , 
               const int *desca , 
               const float *beta , float *c , 
               const int *ic , const int *jc ,
               const int *descc ) {
    pctranu(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void pztranu__(const int *m , const int *n , 
               double *alpha , const double *a , 
               const int *ia , const int *ja , 
               const int *desca , 
               const double *beta , double *c , 
               const int *ic , const int *jc ,
               const int *descc ) {
    pztranu(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

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
            const int *descc ) {
    pstran(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void PDTRAN(const int *m , const int *n , 
            double *alpha , const double *a , 
            const int *ia , const int *ja , 
            const int *desca , 
            const double *beta , double *c , 
            const int *ic , const int *jc ,
            const int *descc ) {
    pdtran(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void PCTRANU(const int *m , const int *n , 
             float *alpha , const float *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const float *beta , float *c , 
             const int *ic , const int *jc ,
             const int *descc ) {
    pctranu(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}

void PZTRANU(const int *m , const int *n , 
             double *alpha , const double *a , 
             const int *ia , const int *ja , 
             const int *desca , 
             const double *beta , double *c , 
             const int *ic , const int *jc ,
             const int *descc ) {
    pztranu(m, n, alpha, a, ia, ja, desca, beta, c, ic, jc, descc);
}
}
