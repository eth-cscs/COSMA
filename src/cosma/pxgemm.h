#pragma once
#include <complex.h>
#ifdef __cplusplus
extern "C" {
#endif
// ScaLAPACK API (override)
void psgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
        float* c, const int* ic, const int* jc, const int* descc);

void pdgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

void pcgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const float _Complex* alpha, const float _Complex* a, const int* ia,
        const int* ja, const int* desca, const float _Complex* b, const int* ib,
        const int* jb, const int* descb, const float _Complex* beta,
        float _Complex* c, const int* ic, const int* jc, const int* descc);

void pzgemm(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
        const double _Complex* alpha, const double _Complex* a, const int* ia,
        const int* ja, const int* desca, const double _Complex* b, const int* ib,
        const int* jb, const int* descb, const double _Complex* beta,
        double _Complex* c, const int* ic, const int* jc, const int* descc);
#ifdef __cplusplus
}
#endif
