#ifdef COSMA_WITH_SCALAPACK
// from std
#include <array>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <complex>
#include <tuple>
#include <vector>

// from cosma
#include <cosma/blacs.hpp>
#include <cosma/pgemm.hpp>

using namespace cosma;

template <typename T>
void fillInt(T &in) {
    std::generate(in.begin(), in.end(), []() { return (int)(10 * drand48()); });
}

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

std::tuple<long, long> run_pdgemm(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Initialize Cblas context
    int ctxt, myid, myrow, mycol, numproc;
    // assume we have 2 x 2 rank grid
    int procrows = 2, proccols = 2;
    blacs::Cblacs_pinfo(&myid, &numproc);
    blacs::Cblacs_get(0, 0, &ctxt);
    char order = 'R';
    blacs::Cblacs_gridinit(&ctxt, &order, procrows, proccols);
    blacs::Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    // describe a problem size
    int m = 10000;
    int n = 10000;
    int k = 10000;

    int bm = 128;
    int bn = 128;
    int bk = 128;

    char trans_a = 'N';
    char trans_b = 'N';
    char trans_c = 'N';

    double alpha = 1.0;
    double beta = 0.0;

    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ic = 1;
    int jc = 1;

    int rsrc = 0;
    int csrc = 0;

    int m_global = m + ia - 1;
    int n_global = n + jb - 1;
    int k_global = k + ja - 1;

    int nrows_a = scalapack::numroc_(&m_global, &bm, &myrow, &rsrc, &procrows);
    int nrows_b = scalapack::numroc_(&k_global, &bk, &myrow, &rsrc, &procrows);
    int nrows_c = scalapack::numroc_(&m_global, &bm, &myrow, &rsrc, &procrows);

    int ncols_a = scalapack::numroc_(&k_global, &bk, &mycol, &csrc, &proccols);
    int ncols_b = scalapack::numroc_(&n_global, &bn, &mycol, &csrc, &proccols);
    int ncols_c = scalapack::numroc_(&n_global, &bn, &mycol, &csrc, &proccols);

    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c(nrows_c * ncols_c);

    // initialize matrices A, B and C
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    scalapack::descinit_(&desc_a[0], &m_global, &k_global, &bm, &bk, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    scalapack::descinit_(&desc_b[0], &k_global, &n_global, &bk, &bn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    scalapack::descinit_(&desc_c[0], &m_global, &n_global, &bm, &bn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    srand48(rank);
    fillInt(a);
    fillInt(b);
    fillInt(c);

    // running COSMA wrapper
    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    cosma::pdgemm(trans_a, trans_b, m, n, k,
           alpha, a.data(), ia, ja, &desc_a[0],
           b.data(), ib, jb, &desc_b[0], beta,
           c.data(), ic, jc, &desc_c[0]);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();

    long cosma_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    // running ScaLAPACK
    MPI_Barrier(comm);
    start = std::chrono::steady_clock::now();
    scalapack::pdgemm_(&trans_a, &trans_b, &m, &n, &k,
           &alpha, a.data(), &ia, &ja, &desc_a[0],
           b.data(), &ib, &jb, &desc_b[0], &beta,
           c.data(), &ic, &jc, &desc_c[0]);
    MPI_Barrier(comm);
    end = std::chrono::steady_clock::now();

    long scalapack_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    blacs::Cblacs_gridexit(ctxt);

    return std::make_tuple(cosma_time, scalapack_time);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_iter = 3;

    std::vector<long> scalapack_times;
    std::vector<long> cosma_times;

    for (int i = 0; i < n_iter; ++i) {
        long cosma_time, scalapack_time;
        std::tie(cosma_time, scalapack_time) = run_pdgemm();
        cosma_times.push_back(cosma_time);
        scalapack_times.push_back(scalapack_time);
    }

    if (rank == 0) {
        std::sort(cosma_times.begin(), cosma_times.end());
        std::sort(scalapack_times.begin(), scalapack_times.end());

        std::cout << "COSMA_PDGEMM TIMES [ms] = ";
        for (auto &time : cosma_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;

        std::cout << "PDGEMM TIMES [ms] = ";
        for (auto &time : scalapack_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
#endif
