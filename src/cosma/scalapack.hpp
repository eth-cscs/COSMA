#ifdef COSMA_WITH_SCALAPACK
#pragma once
#include <mpi.h>
#include "blacs.hpp"
#include <complex>

#ifdef COSMA_WITH_MKL
#include <mkl_scalapack.h>
#else
#include <scalapack.h>
#endif

namespace scalapack {
extern "C" {
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);
    int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

    void psgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
            const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
            float* c, const int* ic, const int* jc, const int* descc);

    void pdgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);

    void pcgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
            const std::complex<float>* alpha, const std::complex<float>* a, const int* ia,
            const int* ja, const int* desca, const std::complex<float>* b, const int* ib,
            const int* jb, const int* descb, const std::complex<float>* beta,
            std::complex<float>* c, const int* ic, const int* jc, const int* descc);

    void pzgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
            const std::complex<double>* alpha, const std::complex<double>* a, const int* ia,
            const int* ja, const int* desca, const std::complex<double>* b, const int* ib,
            const int* jb, const int* descb, const std::complex<double>* beta,
            std::complex<double>* c, const int* ic, const int* jc, const int* descc);
}
}
#endif

