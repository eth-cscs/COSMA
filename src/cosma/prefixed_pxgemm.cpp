#include <cosma/cosma_pxgemm.hpp>

extern "C" {
#include <cosma/prefixed_pxgemm.h>

// scalapack routines that will be invoked if problem too small for COSMA
void psgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void pdgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

void pcgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void pzgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

// Reimplement ScaLAPACK signatures functions
void cosma_pdgemm_(const char* trans_a,
            const char* trans_b,
            const int* m,
            const int* n,
            const int* k,
            const double* alpha,
            const double *a,
            const int* ia,
            const int* ja,
            const int* desca,
            const double *b,
            const int* ib,
            const int* jb,
            const int *descb,
            const double* beta,
            double *c,
            const int* ic,
            const int* jc,
            const int* descc) {
    if (cosma::is_problem_too_small(*m, *n, *k)) {
        pdgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
        return;
    }
    cosma::pxgemm<double>(*trans_a,
                  *trans_b,
                  *m,
                  *n,
                  *k,
                  *alpha,
                  a,
                  *ia,
                  *ja,
                  desca,
                  b,
                  *ib,
                  *jb,
                  descb,
                  *beta,
                  c,
                  *ic,
                  *jc,
                  descc);
}

void cosma_psgemm_(const char* trans_a,
             const char* trans_b,
             const int* m,
             const int* n,
             const int* k,
             const float* alpha,
             const float* a,
             const int* ia,
             const int* ja,
             const int* desca,
             const float* b,
             const int* ib,
             const int* jb,
             const int* descb,
             const float* beta,
             float* c,
             const int* ic,
             const int* jc,
             const int* descc) {
    if (cosma::is_problem_too_small(*m, *n, *k)) {
        psgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
        return;
    }
    cosma::pxgemm<float>(*trans_a,
                 *trans_b,
                 *m,
                 *n,
                 *k,
                 *alpha,
                 a,
                 *ia,
                 *ja,
                 desca,
                 b,
                 *ib,
                 *jb,
                 descb,
                 *beta,
                 c,
                 *ic,
                 *jc,
                 descc);
}

void cosma_pcgemm_(const char* trans_a,
            const char* trans_b,
            const int* m,
            const int* n,
            const int* k,
            const float * alpha,
            const float * a,
            const int* ia,
            const int* ja,
            const int *desca,
            const float * b,
            const int* ib,
            const int* jb,
            const int *descb,
            const float * beta,
            float  *c,
            const int* ic,
            const int* jc,
            const int *descc) {
    if (cosma::is_problem_too_small(*m, *n, *k)) {
        pcgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
        return;
    }

    cosma::pxgemm<std::complex<float>>(*trans_a,
                    *trans_b,
                    *m,
                    *n,
                    *k,
                    reinterpret_cast<const std::complex<float>&>(*alpha),
                    reinterpret_cast<const std::complex<float>*>(a),
                    *ia,
                    *ja,
                    desca,
                    reinterpret_cast<const std::complex<float>*>(b),
                    *ib,
                    *jb,
                    descb,
                    reinterpret_cast<const std::complex<float>&>(*beta),
                    reinterpret_cast<std::complex<float>*>(c),
                    *ic,
                    *jc,
                    descc);
}

void cosma_pzgemm_(const char* trans_a,
            const char* trans_b,
            const int* m,
            const int* n,
            const int* k,
            const double * alpha,
            const double * a,
            const int* ia,
            const int* ja,
            const int *desca,
            const double  *b,
            const int* ib,
            const int* jb,
            const int* descb,
            const double * beta,
            double * c,
            const int* ic,
            const int* jc,
            const int *descc) {
    if (cosma::is_problem_too_small(*m, *n, *k)) {
        pzgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
        return;
    }

    cosma::pxgemm<std::complex<double>>(*trans_a,
                     *trans_b,
                     *m,
                     *n,
                     *k,
                     reinterpret_cast<const std::complex<double>&>(*alpha),
                     reinterpret_cast<const std::complex<double>*>(a),
                     *ia,
                     *ja,
                     desca,
                     reinterpret_cast<const std::complex<double>*>(b),
                     *ib,
                     *jb,
                     descb,
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

void cosma_psgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    cosma_psgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void cosma_pdgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    cosma_pdgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void cosma_pcgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc) {
    cosma_pcgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void cosma_pzgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc) {
    cosma_pzgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

// *********************************************************************************
// Same as previously, but with double underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void COSMA_PSGEMM_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    cosma_psgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void COSMA_PDGEMM_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    cosma_pdgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void COSMA_PCGEMM_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc) {
    cosma_pcgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void COSMA_PZGEMM_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc) {
    cosma_pzgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void COSMA_PSGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    cosma_psgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void COSMA_PDGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    cosma_pdgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void COSMA_PCGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc) {
    cosma_pcgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void COSMA_PZGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc) {
    cosma_pzgemm_(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}
}
