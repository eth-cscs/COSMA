#include <cosma/cosma_pxgemm.hpp>
#include <cosma/environment_variables.hpp>

extern "C" {
#include <cosma/pxgemm.h>
#include "interpose.h"

bool is_problem_too_small(int m, int n, int k) {
    static const int cosma_dim_threshold = cosma::get_cosma_dim_threshold();
    return std::min(m, std::min(n, k)) < cosma_dim_threshold;
}

// Reimplement ScaLAPACK signatures functions
INTERPOSE_C_VOID(psgemm_,
        (const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc),
        (trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc)
        ) {
    if (is_problem_too_small(*m, *n, *k)) {
        Real__psgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
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

INTERPOSE_C_VOID(pdgemm_, 
        (const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc),
        (trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc)
        ){
    if (is_problem_too_small(*m, *n, *k)) {
        Real__pdgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
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

INTERPOSE_C_VOID(pcgemm_, 
        (const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc),
        (trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc)
        ){
    if (is_problem_too_small(*m, *n, *k)) {
        Real__pcgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
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

INTERPOSE_C_VOID(pzgemm_, 
        (const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc),
        (trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc)
        ){
    if (is_problem_too_small(*m, *n, *k)) {
        Real__pzgemm_(trans_a, trans_b, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
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
void psgemm(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    psgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pdgemm(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    pdgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pcgemm(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc) {
    pcgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pzgemm(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc) {
    pzgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

// *********************************************************************************
// Same as previously, but with double underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void PSGEMM_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    psgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PDGEMM_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    pdgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PCGEMM_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc) {
    pcgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PZGEMM_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc) {
    pzgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void PSGEMM(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    psgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PDGEMM(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    pdgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PCGEMM(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc) {
    pcgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PZGEMM(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc) {
    pzgemm_(trans_a, trans_b, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}
}
