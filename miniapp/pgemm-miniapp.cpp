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

// runs cosma pdgemm wrapper for n_iter times and returns
// a vector of timings (in milliseconds) of size n_iter
std::vector<long> run_pdgemm(int m, int n, int k,
        int bm, int bn, int bk,
        int p, int q, 
        int rank, int n_iter,
        MPI_Comm comm = MPI_COMM_WORLD) {

    // ***********************************
    //   Cblacs context initialization
    // ***********************************
    int ctxt, myrow, mycol;
    blacs::Cblacs_get(0, 0, &ctxt);
    char order = 'R';
    blacs::Cblacs_gridinit(&ctxt, &order, p, q);
    blacs::Cblacs_pcoord(ctxt, rank, &myrow, &mycol);

    // ***********************************
    //   describe the problem parameters
    // ***********************************
    // transpose flags
    char trans_a = 'N';
    char trans_b = 'N';
    char trans_c = 'N';

    double alpha = 1.0;
    double beta = 0.0;

    // start indices of submatrices for multiplication
    // matrix A
    int ia = 1;
    int ja = 1;

    // matrix B
    int ib = 1;
    int jb = 1;

    // matrix C
    int ic = 1;
    int jc = 1;

    // rank source parameters (row and column of a rank
    // in a rank grid that owns the first row of matrices
    int rsrc = 0;
    int csrc = 0;

    // global problem size
    // m, n, k are just sizes that we want to multiply
    // starting from (ia-1, ja-1), (ib-1, jb-1) and (ic-1, jc-1)
    // this makes the global problem size m+ia-1, n+jb-1, k+ja-1
    int m_global = m + ia - 1;
    int n_global = n + jb - 1;
    int k_global = k + ja - 1;

    // ********************************************
    //   allocate scalapack buffers for matrices
    // ********************************************
    // get the local number of rows that this rank owns
    int nrows_a = scalapack::numroc_(&m_global, &bm, &myrow, &rsrc, &p);
    int nrows_b = scalapack::numroc_(&k_global, &bk, &myrow, &rsrc, &p);
    int nrows_c = scalapack::numroc_(&m_global, &bm, &myrow, &rsrc, &p);

    // get the local number of cols that this rank owns
    int ncols_a = scalapack::numroc_(&k_global, &bk, &mycol, &csrc, &q);
    int ncols_b = scalapack::numroc_(&n_global, &bn, &mycol, &csrc, &q);
    int ncols_c = scalapack::numroc_(&n_global, &bn, &mycol, &csrc, &q);

    // allocate size for the local buffers
    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c(nrows_c * ncols_c);

    // initialize descriptors for matrices A, B and C
    // use scalapack routine descinit_
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    scalapack::descinit_(&desc_a[0], &m_global, &k_global, &bm, &bk, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    scalapack::descinit_(&desc_b[0], &k_global, &n_global, &bk, &bn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    scalapack::descinit_(&desc_c[0], &m_global, &n_global, &bm, &bn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    srand48(rank);
    fillInt(c);

    // vectors to store timings
    std::vector<long> cosma_times(n_iter);

    // ***********************************
    //   performing the multiplication
    // ***********************************
    // run COSMA pdgemm wrapper n_iter times
    for (int i = 0; i < n_iter; ++i) {
        // refill matrices with random data to avoid
        // reusing the cache in subsequent iterations
        fillInt(a);
        fillInt(b);

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
        cosma_times[i] = cosma_time;
    }

    // exit blacs context
    blacs::Cblacs_gridexit(ctxt);

    // sort cosma timings in increasing order
    std::sort(cosma_times.begin(), cosma_times.end());

    return cosma_times;
}

int main(int argc, char **argv) {
    // **************************************
    //   setup MPI and command-line parser
    // **************************************
    options::initialize(argc, argv);

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // **************************************
    //   readout the command line arguments
    // **************************************
    auto m = options::next_int("-m", "--m_dim", "number of rows of A and C.", 1000);
    auto n = options::next_int("-n", "--n_dim", "number of columns of B and C.", 1000);
    auto k = options::next_int("-k", "--k_dim", "number of columns of A and rows of B.", 1000);

    auto bm = options::next_int("-bm", "--m_block", "block size for the number of rows of A and C.", 128);
    auto bn = options::next_int("-bn", "--n_block", "block size for the number of columns of B and C.", 128);
    auto bk = options::next_int("-bk", "--k_block", "block size for the number of columns of A and rows of B.", 128);

    auto p = options::next_int("-p", "--p_row", "number of rows in a processor grid.", 1);
    auto q = options::next_int("-q", "--q_row", "number of columns in a processor grid.", P);

    if (p * q != P) {
        std::runtime_error("Number of processors in a grid has to match the number of available ranks.");
    }

    int n_iter = 3;

    // *******************************
    //   perform the multiplication
    // ******************************
    std::vector<long> cosma_times = 
        run_pdgemm(m, n, k, bm, bn, bk, p, q, rank, n_iter);

    // *****************
    //   output times
    // *****************
    if (rank == 0) {
        std::cout << "COSMA_PDGEMM TIMES [ms] = ";
        for (auto &time : cosma_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
