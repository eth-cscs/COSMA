#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// scalapack api
void psgemr2d(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void pdgemr2d(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void pcgemr2d(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void pzgemr2d(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void pigemr2d(const int *m, const int *n,
              const int *a,
              const int *ia, const int *ja,
              const int *desca,
              int *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

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
               const int *ictxt);

void pdgemr2d_(const int *m, const int *n,
               const double *a,
               const int *ia, const int *ja,
               const int *desca,
               double *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt);

void pcgemr2d_(const int *m, const int *n,
               const float *a,
               const int *ia, const int *ja,
               const int *desca,
               float *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt);

void pzgemr2d_(const int *m, const int *n,
               const double *a,
               const int *ia, const int *ja,
               const int *desca,
               double *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt);

void pigemr2d_(const int *m, const int *n,
               const int *a,
               const int *ia, const int *ja,
               const int *desca,
               int *b,
               const int *ib, const int *jb,
               const int *descb,
               const int *ictxt);

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
                const int *ictxt);

void pdgemr2d__(const int *m, const int *n,
                const double *a,
                const int *ia, const int *ja,
                const int *desca,
                double *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt);

void pcgemr2d__(const int *m, const int *n,
                const float *a,
                const int *ia, const int *ja,
                const int *desca,
                float *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt);

void pzgemr2d__(const int *m, const int *n,
                const double *a,
                const int *ia, const int *ja,
                const int *desca,
                double *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt);

void pigemr2d__(const int *m, const int *n,
                const int *a,
                const int *ia, const int *ja,
                const int *desca,
                int *b,
                const int *ib, const int *jb,
                const int *descb,
                const int *ictxt);

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
              const int *ictxt);

void PDGEMR2D(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void PCGEMR2D(const int *m, const int *n,
              const float *a,
              const int *ia, const int *ja,
              const int *desca,
              float *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void PZGEMR2D(const int *m, const int *n,
              const double *a,
              const int *ia, const int *ja,
              const int *desca,
              double *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

void PIGEMR2D(const int *m, const int *n,
              const int *a,
              const int *ia, const int *ja,
              const int *desca,
              int *b,
              const int *ib, const int *jb,
              const int *descb,
              const int *ictxt);

#ifdef __cplusplus
}
#endif
