#pragma once
/*
 * This is a COSMA backend for matrices given in ScaLAPACK format.
 * It is less efficient than using cosma::multiply directly with COSMA data layout.
 * Thus, here we pay the price of transforming matrices between scalapack and COSMA layout.
 */
namespace cosma {
using zdouble_t = std::complex<double>;
using zfloat_t = std::complex<float>;

// TODO: alpha ignored at the moment
template <typename T>
void pgemm(const char trans_a, const char trans_b, const int m, const int n, const int k,
           const T alpha, const T* a, const int ia, const int ja, const int* desca,
           const T* b, const int ib, const int jb, const int* descb, const T beta,
           T* c, const int ic, const int jc, const int* descc);

// ScaLAPACK signatures (override)
void pdgemm(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const double alpha, const double* a, const int ia, const int ja, const int* desca,
    const double* b, const int ib, const int jb, const int* descb, const double beta,
    double* c, const int ic, const int jc, const int* descc);

void psgemm(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const float alpha, const float* a, const int ia, const int ja, const int* desca,
    const float* b, const int ib, const int jb, const int* descb, const float beta,
    float* c, const int ic, const int jc, const int* descc);

void pcgemm(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const zfloat_t alpha, const zfloat_t* a, const int ia, const int ja, const int* desca,
    const zfloat_t* b, const int ib, const int jb, const int* descb, const zfloat_t beta,
    zfloat_t* c, const int ic, const int jc, const int* descc);

void pzgemm(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const zdouble_t alpha, const zdouble_t* a, const int ia, const int ja, const int* desca,
    const zdouble_t* b, const int ib, const int jb, const int* descb, const zdouble_t beta,
    zdouble_t* c, const int ic, const int jc, const int* descc);
}
