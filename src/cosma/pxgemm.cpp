#include <cosma/pgemm.hpp>
#include <cosma/pxgemm.h>
#include <mpi.h>
#include <stdio.h>

extern "C" {
// Reimplement ScaLAPACK signatures functions
void pdgemm(const char* trans_a,
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

    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) {
    //     printf("m = %d, n = %d, k = %d", *m, *n, *k);
    //     fflush(stdout);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    cosma::pgemm<double>(*trans_a,
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

void psgemm(const char* trans_a,
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
    cosma::pgemm<float>(*trans_a,
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

void pcgemm(const char* trans_a,
            const char* trans_b,
            const int* m,
            const int* n,
            const int* k,
            const float _Complex* alpha,
            const float _Complex* a,
            const int* ia,
            const int* ja,
            const int *desca,
            const float _Complex* b,
            const int* ib,
            const int* jb,
            const int *descb,
            const float _Complex* beta,
            float _Complex *c,
            const int* ic,
            const int* jc,
            const int *descc) {

    cosma::pgemm<std::complex<float>>(*trans_a,
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

void pzgemm(const char* trans_a,
            const char* trans_b,
            const int* m,
            const int* n,
            const int* k,
            const double _Complex* alpha,
            const double _Complex* a,
            const int* ia,
            const int* ja,
            const int *desca,
            const double _Complex *b,
            const int* ib,
            const int* jb,
            const int* descb,
            const double _Complex* beta,
            double _Complex* c,
            const int* ic,
            const int* jc,
            const int *descc) {

    cosma::pgemm<std::complex<double>>(*trans_a,
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

void psgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    psgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pdgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    pdgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pcgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float _Complex* alpha, const float _Complex* a, const int* ia,
        const int* ja, const int* desca, const float _Complex* b, const int* ib,
        const int* jb, const int* descb, const float _Complex* beta,
        float _Complex* c, const int* ic, const int* jc, const int* descc) {
    pcgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pzgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double _Complex* alpha, const double _Complex* a, const int* ia,
        const int* ja, const int* desca, const double _Complex* b, const int* ib,
        const int* jb, const int* descb, const double _Complex* beta,
        double _Complex* c, const int* ic, const int* jc, const int* descc) {
    pzgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

// *********************************************************************************
// Same as previously, but with double underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void psgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    psgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pdgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    pdgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pcgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float _Complex* alpha, const float _Complex* a, const int* ia,
        const int* ja, const int* desca, const float _Complex* b, const int* ib,
        const int* jb, const int* descb, const float _Complex* beta,
        float _Complex* c, const int* ic, const int* jc, const int* descc) {
    pcgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void pzgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double _Complex* alpha, const double _Complex* a, const int* ia,
        const int* ja, const int* desca, const double _Complex* b, const int* ib,
        const int* jb, const int* descb, const double _Complex* beta,
        double _Complex* c, const int* ic, const int* jc, const int* descc) {
    pzgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void PSGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc) {
    psgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PDGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc) {
    pdgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PCGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float _Complex* alpha, const float _Complex* a, const int* ia,
        const int* ja, const int* desca, const float _Complex* b, const int* ib,
        const int* jb, const int* descb, const float _Complex* beta,
        float _Complex* c, const int* ic, const int* jc, const int* descc) {
    pcgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}

void PZGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double _Complex* alpha, const double _Complex* a, const int* ia,
        const int* ja, const int* desca, const double _Complex* b, const int* ib,
        const int* jb, const int* descb, const double _Complex* beta,
        double _Complex* c, const int* ic, const int* jc, const int* descc) {
    pzgemm(trans_a, transb, m, n, k,
           alpha, a, ia, ja, desca,
           b, ib, jb, descb,
           beta, c, ic, jc, descc);
}
}
