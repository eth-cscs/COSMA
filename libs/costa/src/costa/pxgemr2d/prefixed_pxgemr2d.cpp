#include <costa/pxgemr2d/costa_pxgemr2d.hpp>

extern "C" {
#include <costa/pxgemr2d/prefixed_pxgemr2d.h>

void costa_pigemr2d(const int *m, const int *n,
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

void costa_psgemr2d(const int *m, const int *n,
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

void costa_pdgemr2d(const int *m, const int *n,
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

void costa_pcgemr2d(const int *m, const int *n,
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

void costa_pzgemr2d(const int *m, const int *n,
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
void costa_psgemr2d_(const int *m, const int *n,
               const float *a,
               const int *ia, const int *ja,
               const int *desca,
               float *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    costa_psgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pdgemr2d_(const int *m, const int *n,
               const double *a,
               const int *ia, const int *ja,
               const int *desca,
               double *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    costa_pdgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pcgemr2d_(const int *m, const int *n,
               const float *a,
               const int *ia, const int *ja,
               const int *desca,
               float *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    costa_pcgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pzgemr2d_(const int *m, const int *n,
               const double *a,
               const int *ia, const int *ja,
               const int *desca,
               double *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    costa_pzgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pigemr2d_(const int *m, const int *n,
               const int *a,
               const int *ia, const int *ja,
               const int *desca,
               int *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt) {
    costa_pigemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

// *********************************************************************************
// Same as previously, but with added double underscores at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void costa_psgemr2d__(const int *m, const int *n,
                const float *a,
                const int *ia, const int *ja,
                const int *desca,
                float *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    costa_psgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pdgemr2d__(const int *m, const int *n,
                const double *a,
                const int *ia, const int *ja,
                const int *desca,
                double *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    costa_pdgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pcgemr2d__(const int *m, const int *n,
                const float *a,
                const int *ia, const int *ja,
                const int *desca,
                float *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    costa_pcgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pzgemr2d__(const int *m, const int *n,
                const double *a,
                const int *ia, const int *ja,
                const int *desca,
                double *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    costa_pzgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void costa_pigemr2d__(const int *m, const int *n,
                const int *a,
                const int *ia, const int *ja,
                const int *desca,
                int *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt) {
    costa_pigemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************
void COSTA_PSGEMR2D(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    costa_psgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void COSTA_PDGEMR2D(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    costa_pdgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void COSTA_PCGEMR2D(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    costa_pcgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void COSTA_PZGEMR2D(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    costa_pzgemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

void COSTA_PIGEMR2D(const int *m, const int *n,
              const int *a,
              const int *ia, const int *ja,
              const int *desca,
              int *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt) {
    costa_pigemr2d(m, n, a, ia, ja, desca, b, ib, jb, descb, ictxt);
}

}
