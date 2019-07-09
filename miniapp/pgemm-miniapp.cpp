// from std
#include <array>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <complex>
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

/*
long run_scalapack(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Initialize Cblas context
    int ctxt, myid, myrow, mycol, numproc;
    // assume we have 2x2 processor grid
    int procrows = 2, proccols = 2;
    std::cout << "Cblacs_pinfo" << std::endl;
    Cblacs_pinfo(&myid, &numproc);
    std::cout << "Cblacs_get" << std::endl;
    Cblacs_get(0, 0, &ctxt);
    char order = 'R';
    std::cout << "Cblacs_gridinit" << std::endl;
    Cblacs_gridinit(&ctxt, &order, procrows, proccols);
    std::cout << "Cblacs_pcoord" << std::endl;
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    std::cout << "Rank = " << rank << ", myid = " << myid << std::endl;
    std::cout << "My pcoord = " << myrow << ", " << mycol << std::endl;

    // describe a problem size
    int m = 10;
    int n = 10;
    int k = 10;

    int bm = 2;
    int bn = 2;
    int bk = 2;

    char trans_a = 'N';
    char trans_b = 'N';
    char trans_c = 'N';

    int alpha = 1.0;
    int beta = 0.0;

    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ic = 1;
    int jc = 1;

    int rsrc = 0;
    int csrc = 0;

    int iZERO = 0;

    std::cout << "numroc A, rows" << std::endl;
    int nrows_a = get_numroc(m+ia-1, bm, myrow, rsrc, procrows);
    std::cout << "numroc B, rows" << std::endl;
    int nrows_b = get_numroc(k+ib-1, bk, myrow, rsrc, procrows);
    std::cout << "numroc C, rows" << std::endl;
    int nrows_c = get_numroc(m+ic-1, bm, myrow, rsrc, procrows);

    std::cout << "numroc A, cols" << std::endl;
    int ncols_a = get_numroc(k+ja-1, bk, mycol, csrc, proccols);
    std::cout << "numroc B, cols" << std::endl;
    int ncols_b = get_numroc(n+jb-1, bn, mycol, csrc, proccols);
    std::cout << "numroc C, cols" << std::endl;
    int ncols_c = get_numroc(n+jc-1, bn, mycol, csrc, proccols);

    std::cout << "Initializing ScaLAPACK buffers for A, B and C" << std::endl;
    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c(nrows_c * ncols_c);

    // initialize matrices A, B and C
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    std::cout << "descinit A" << std::endl;
    descinit(&desc_a[0], &m, &k, &bm, &bk, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    std::cout << "descinit B" << std::endl;
    descinit(&desc_b[0], &k, &n, &bk, &bn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    std::cout << "descinit C" << std::endl;
    descinit(&desc_c[0], &m, &n, &bm, &bn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    std::cout << "Filling up ScaLAPACK matrices with random data." << std::endl;
    srand48(rank);
    fillInt(a);
    fillInt(b);
    fillInt(c);

    std::cout << "Invoking pdgemm_wrapper" << std::endl;
    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    cosma::pdgemm(trans_a, trans_b, m, n, k,
           alpha, a.data(), ia, ja, &desc_a[0],
           b.data(), ib, jb, &desc_b[0], beta,
           c.data(), ic, jc, &desc_c[0]);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Finished pdgemm_wrapper" << std::endl;

    std::cout << "Cblacs_gridexit" << std::endl;
    Cblacs_gridexit(ctxt);
    std::cout << "Cblacs_exit" << std::endl;

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}
*/

long run_cosma_with_scalapack(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Initialize Cblas context */
    int ctxt, myid, myrow, mycol, numproc;
    // assume we have 2x2 processor grid
    int procrows = 2, proccols = 2;
    std::cout << "Cblacs_pinfo" << std::endl;
    blacs::Cblacs_pinfo(&myid, &numproc);
    std::cout << "Cblacs_get" << std::endl;
    blacs::Cblacs_get(0, 0, &ctxt);
    char order = 'R';
    std::cout << "Cblacs_gridinit" << std::endl;
    blacs::Cblacs_gridinit(&ctxt, &order, procrows, proccols);
    std::cout << "Cblacs_pcoord" << std::endl;
    blacs::Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    std::cout << "Rank = " << rank << ", myid = " << myid << std::endl;
    std::cout << "My pcoord = " << myrow << ", " << mycol << std::endl;

    // describe a problem size
    int m = 10;
    int n = 10;
    int k = 10;

    int bm = 2;
    int bn = 2;
    int bk = 2;

    char trans_a = 'N';
    char trans_b = 'N';
    char trans_c = 'N';

    int alpha = 1.0;
    int beta = 0.0;

    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ic = 1;
    int jc = 1;

    int rsrc = 0;
    int csrc = 0;

    int iZERO = 0;

    int m_global = m + ia - 1;
    int n_global = n + jb - 1;
    int k_global = k + ja - 1;

    std::cout << "numroc A, rows" << std::endl;
    int nrows_a = scalapack::numroc_(&m_global, &m, &myrow, &rsrc, &procrows);
    std::cout << "numroc B, rows" << std::endl;
    int nrows_b = scalapack::numroc_(&k_global, &bk, &myrow, &rsrc, &procrows);
    std::cout << "numroc C, rows" << std::endl;
    int nrows_c = scalapack::numroc_(&m_global, &bm, &myrow, &rsrc, &procrows);

    std::cout << "numroc A, cols" << std::endl;
    int ncols_a = scalapack::numroc_(&k_global, &bk, &mycol, &csrc, &proccols);
    std::cout << "numroc B, cols" << std::endl;
    int ncols_b = scalapack::numroc_(&n_global, &bn, &mycol, &csrc, &proccols);
    std::cout << "numroc C, cols" << std::endl;
    int ncols_c = scalapack::numroc_(&n_global, &bn, &mycol, &csrc, &proccols);

    std::cout << "Initializing ScaLAPACK buffers for A, B and C" << std::endl;
    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c(nrows_c * ncols_c);

    // initialize matrices A, B and C
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    std::cout << "descinit A" << std::endl;
    scalapack::descinit_(&desc_a[0], &m_global, &k_global, &bm, &bk, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    std::cout << "descinit B" << std::endl;
    scalapack::descinit_(&desc_b[0], &k_global, &n_global, &bk, &bn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    std::cout << "descinit C" << std::endl;
    scalapack::descinit_(&desc_c[0], &m_global, &n_global, &bm, &bn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    std::cout << "Filling up ScaLAPACK matrices with random data." << std::endl;
    srand48(rank);
    fillInt(a);
    fillInt(b);
    fillInt(c);

    std::cout << "Invoking pdgemm_wrapper" << std::endl;
    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    pgemm<double>(trans_a, trans_b, m, n, k,
           alpha, a.data(), ia, ja, &desc_a[0],
           b.data(), ib, jb, &desc_b[0], beta,
           c.data(), ic, jc, &desc_c[0]);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Finished pdgemm_wrapper" << std::endl;

    std::cout << "Cblacs_gridexit" << std::endl;
    blacs::Cblacs_gridexit(ctxt);
    std::cout << "Cblacs_exit" << std::endl;

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    std::cout << "Initialized MPI." << std::endl;

    int n_iter = 3;

    std::vector<long> cosma_with_scalapack_times;
    for (int i = 0; i < n_iter; ++i) {
        long t_run = 0;
        t_run = run_cosma_with_scalapack();
        cosma_with_scalapack_times.push_back(t_run);
    }
    std::sort(cosma_with_scalapack_times.begin(), cosma_with_scalapack_times.end());

    // std::vector<long> scalapack_times;
    // for (int i = 0; i < n_iter; ++i) {
    //     long t_run = 0;
    //     t_run = run_scalapack();
    //     scalapack_times.push_back(t_run);
    // }
    // std::sort(scalapack_times.begin(), scalapack_times.end());

    std::cout << "COSMA_PDGEMM_WRAPPER TIMES [ms] = ";
    for (auto &time : cosma_with_scalapack_times) {
        std::cout << time << " ";
    }
    std::cout << std::endl;

    // std::cout << "PDGEMM TIMES [ms] = ";
    // for (auto &time : scalapack_times) {
    //     std::cout << time << " ";
    // }
    // std::cout << std::endl;

    MPI_Finalize();

    return 0;
}
