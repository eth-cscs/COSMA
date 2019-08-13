// from std
#include <array>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <complex>
#include <tuple>
#include <vector>
#include <stdexcept>

// from cosma
#include <cosma/multiply.hpp>
#include <cosma/blacs.hpp>
#include <cosma/pgemm.hpp>

// from options
#include <options.hpp>

using namespace cosma;

// random number generator
// we cast them to ints, so that we can more easily test them
// but it's not necessary (they are anyway stored as double's)
template <typename T>
void fillInt(T &in) {
    std::generate(in.begin(), in.end(), []() { return (int)(10 * drand48()); });
}

// **********************
//   ScaLAPACK routines
// **********************
namespace scalapack {
extern "C" {
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);
    int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

    void pdgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);
}
}

// compares two vectors up to eps precision, returns true if they are equal
bool validate_results(std::vector<double>& v1, std::vector<double>& v2, double eps=1e-5) {
    int size = std::min(v1.size(), v2.size());
    for (size_t i = 0; i < size; ++i) {
        if (v1[i] != v2[i]) {
            return false;
        }
    }
    return true;
}

int transpose_if(char transpose_flag, int row, int col) {
    return transpose_flag == 'N' ? row : col;
}

// runs cosma or scalapack pdgemm wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
bool test_pdgemm(int m, int n, int k, // matrix sizes
        int block_m, int block_n, int block_k, // blocks sizes
        int sub_m, int sub_n, int sub_k, // defines submatrices
        char trans_a, char trans_b, // transpose flags
        int p, int q, // processor grid
        double alpha, double beta, // processor grid
        int rank, MPI_Comm comm) {
    // ***********************************
    //   Cblacs context initialization
    // ***********************************
    int myrow, mycol, ctxt;
    char order = 'R';
    ctxt = blacs::Csys2blacs_handle(comm);
    blacs::Cblacs_gridinit(&ctxt, &order, p, q);
    blacs::Cblacs_pcoord(ctxt, rank, &myrow, &mycol);

    // ***********************************
    //   describe the problem parameters
    // ***********************************
    // start indices of submatrices for multiplication
    // matrix A
    int ia = transpose_if(trans_a, sub_m, sub_k);
    int ja = transpose_if(trans_a, sub_k, sub_m);

    // matrix B
    int ib = transpose_if(trans_b, sub_k, sub_n);
    int jb = transpose_if(trans_b, sub_n, sub_k);

    // matrix C
    int ic = sub_m;
    int jc = sub_n;

    // rank source parameters (row and column of a rank
    // in a rank grid that owns the first row of matrices
    int rsrc = 0;
    int csrc = 0;

    // global problem size
    // m, n, k are just sizes that we want to multiply
    // starting from (ia-1, ja-1), (ib-1, jb-1) and (ic-1, jc-1)
    // this makes the global problem size m+ia-1, n+jb-1, k+ja-1
    int am = transpose_if(trans_a, m, k) + ia - 1;
    int an = transpose_if(trans_a, k, m) + ja - 1;
    int bm = transpose_if(trans_b, k, n) + ib - 1;
    int bn = transpose_if(trans_b, n, k) + jb - 1;
    int cm = m + ic - 1;
    int cn = n + jc - 1;

    // This is for compatible blocks
    // in general p*gemm works for any combination of them
    int bam = transpose_if(trans_a, block_m, block_k);
    int ban = transpose_if(trans_a, block_k, block_m);
    int bbm = transpose_if(trans_b, block_k, block_n);
    int bbn = transpose_if(trans_b, block_n, block_k);
    int bcm = block_m;
    int bcn = block_n;

    // ********************************************
    //   allocate scalapack buffers for matrices
    // ********************************************
    // get the local number of rows that this rank owns
    int nrows_a = scalapack::numroc_(&am, &bam, &myrow, &rsrc, &p);
    int nrows_b = scalapack::numroc_(&bm, &bbm, &myrow, &rsrc, &p);
    int nrows_c = scalapack::numroc_(&cm, &bcm, &myrow, &rsrc, &p);

    // get the local number of cols that this rank owns
    int ncols_a = scalapack::numroc_(&an, &ban, &mycol, &csrc, &q);
    int ncols_b = scalapack::numroc_(&bn, &bbn, &mycol, &csrc, &q);
    int ncols_c = scalapack::numroc_(&cn, &bcn, &mycol, &csrc, &q);

    // allocate size for the local buffers
    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c_scalapack(nrows_c * ncols_c);
    std::vector<double> c_cosma(nrows_c * ncols_c);

    // initialize descriptors for matrices A, B and C
    // use scalapack routine descinit_
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    scalapack::descinit_(&desc_a[0], &am, &an, &bam, &ban, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    scalapack::descinit_(&desc_b[0], &bm, &bn, &bbm, &bbn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    scalapack::descinit_(&desc_c[0], &cm, &cn, &bcm, &bcn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    srand48(rank);

    fillInt(a);
    fillInt(b);
    fillInt(c_cosma);
    fillInt(c_scalapack);

    // ***********************************
    //          run COSMA PDGEMM
    // ***********************************
    // running COSMA wrapper
    cosma::pdgemm(trans_a, trans_b, m, n, k,
           alpha, a.data(), ia, ja, &desc_a[0],
           b.data(), ib, jb, &desc_b[0], beta,
           c_cosma.data(), ic, jc, &desc_c[0]);

    // ***********************************
    //       run ScaLAPACK PDGEMM
    // ***********************************
    // running ScaLAPACK
    scalapack::pdgemm_(&trans_a, &trans_b, &m, &n, &k,
           &alpha, a.data(), &ia, &ja, &desc_a[0],
           b.data(), &ib, &jb, &desc_b[0], &beta,
           c_scalapack.data(), &ic, &jc, &desc_c[0]);


    // if (myrow == 0 && mycol == 0) {
    //     std::cout << "c(cosma) = ";
    //     for (int i = 0; i < c_cosma.size(); ++i) {
    //         std::cout << c_cosma[i] << ", ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "c(scalapack) = ";
    //     for (int i = 0; i < c_scalapack.size(); ++i) {
    //         std::cout << c_scalapack[i] << ", ";
    //     }
    //     std::cout << std::endl;
    // }

    // exit blacs context
    blacs::Cblacs_gridexit(ctxt);

    return validate_results(c_cosma, c_scalapack);
}
