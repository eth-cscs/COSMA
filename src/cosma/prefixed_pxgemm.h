#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// ScaLAPACK API with COSMA prefix
void cosma_psgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void cosma_pdgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

void cosma_pcgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc);

void cosma_pzgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc);

// *********************************************************************************
// Same as previously, but with added underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void cosma_psgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void cosma_pdgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

void cosma_pcgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc);

void cosma_pzgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc);

// *********************************************************************************
// Same as previously, but with double underscore at the end.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void cosma_psgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void cosma_pdgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

void cosma_pcgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc);

void cosma_pzgemm__(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc);

// *********************************************************************************
// Same as previously, but CAPITALIZED.
// This is used for fortran interfaces, in case fortran expects these symbols
// *********************************************************************************

void COSMA_PSGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void COSMA_PDGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

void COSMA_PCGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float * alpha, const float * a, const int* ia,
        const int* ja, const int* desca, const float * b, const int* ib,
        const int* jb, const int* descb, const float * beta,
        float * c, const int* ic, const int* jc, const int* descc);

void COSMA_PZGEMM(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double * alpha, const double * a, const int* ia,
        const int* ja, const int* desca, const double * b, const int* ib,
        const int* jb, const int* descb, const double * beta,
        double * c, const int* ic, const int* jc, const int* descc);

#ifdef __cplusplus
}
#endif
