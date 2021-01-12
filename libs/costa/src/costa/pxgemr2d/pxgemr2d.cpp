#include <costa/pxgemr2d/costa_pxgemr2d.hpp>

extern "C" {
#include <costa/pxgemr2d/pxgemr2d.h>

void pigemr2d(const int *m, const int *n,
              const int *a,
              const int *ia, const int *ja,
              const int *desca,
              int *c,
              const int *ic, const int *jc,
              const int *descc,
              const int *ictxt) {
    costa::pxgemr2d<int>(
                  *m,
                  *n,
                  a,
                  *ia,
                  *ja,
                  desca,
                  c,
                  *ic,
                  *jc,
                  descc,
                  *ictxt);
}

void psgemr2d(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *c,
              const int *ic, const int *jc,
              const int *descc,
              const int *ictxt) {
    costa::pxgemr2d<float>(
                  *m,
                  *n,
                  a,
                  *ia,
                  *ja,
                  desca,
                  c,
                  *ic,
                  *jc,
                  descc,
                  *ictxt);
}

void pdgemr2d(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *c,
              const int *ic, const int *jc,
              const int *descc,
              const int *ictxt) {
    costa::pxgemr2d<double>(
                  *m,
                  *n,
                  a,
                  *ia,
                  *ja,
                  desca,
                  c,
                  *ic,
                  *jc,
                  descc,
                  *ictxt);
}

void pcgemr2d(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *c,
              const int *ic, const int *jc,
              const int *descc,
              const int *ictxt) {
    costa::pxgemr2d<std::complex<float>>(
                  *m,
                  *n,
                  reinterpret_cast<const std::complex<float>*>(a),
                  *ia,
                  *ja,
                  desca,
                  reinterpret_cast<std::complex<float>*>(c),
                  *ic,
                  *jc,
                  descc,
                  *ictxt);
}

void pzgemr2d(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *c,
              const int *ic, const int *jc,
              const int *descc,
              const int *ictxt) {
    costa::pxgemr2d<std::complex<double>>(
                  *m,
                  *n,
                  reinterpret_cast<const std::complex<double>*>(a),
                  *ia,
                  *ja,
                  desca,
                  reinterpret_cast<std::complex<double>*>(c),
                  *ic,
                  *jc,
                  descc,
                  *ictxt);
}

// *********************************************************************************
// Same as previously, but with added underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void psgemr2d_(const int *m, const int *n,
               const float *a,
               const int *ia, const int *ja,
               const int *desca,
               float *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    psgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pdgemr2d_(const int *m, const int *n,
               const double *a,
               const int *ia, const int *ja,
               const int *desca,
               double *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    pdgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pcgemr2d_(const int *m, const int *n,
               const float *a,
               const int *ia, const int *ja,
               const int *desca,
               float *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    pcgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pzgemr2d_(const int *m, const int *n,
               const double *a,
               const int *ia, const int *ja,
               const int *desca,
               double *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    pzgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pigemr2d_(const int *m, const int *n,
               const int *a,
               const int *ia, const int *ja,
               const int *desca,
               int *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    pigemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

// *********************************************************************************
// Same as previously, but with added double underscores at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void psgemr2d__(const int *m, const int *n,
                const float *a,
                const int *ia, const int *ja,
                const int *desca,
                float *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    psgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pdgemr2d__(const int *m, const int *n,
                const double *a,
                const int *ia, const int *ja,
                const int *desca,
                double *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    pdgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pcgemr2d__(const int *m, const int *n,
                const float *a,
                const int *ia, const int *ja,
                const int *desca,
                float *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    pcgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pzgemr2d__(const int *m, const int *n,
                const double *a,
                const int *ia, const int *ja,
                const int *desca,
                double *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    pzgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void pigemr2d__(const int *m, const int *n,
                const int *a,
                const int *ia, const int *ja,
                const int *desca,
                int *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    pigemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void PSGEMR2D(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    psgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void PDGEMR2D(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    pdgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void PCGEMR2D(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    pcgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void PZGEMR2D(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    pzgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void PIGEMR2D(const int *m, const int *n,
              const int *a,
              const int *ia, const int *ja,
              const int *desca,
              int *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    pigemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}
}
