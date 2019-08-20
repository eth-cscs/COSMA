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
#include <cosma/context.hpp>
#include <cosma/profiler.hpp>

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

    void pdgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);
}
}

int transpose_if(char transpose_flag, int row, int col) {
    return transpose_flag == 'N' ? row : col;
}

std::pair<int, int> transpose_if(char transpose_flag, std::pair<int, int> p) {
    std::pair<int, int> transposed{p.second, p.first};
    return transpose_flag == 'T' ? transposed : p;
}

// runs cosma or scalapack pdgemm wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
std::vector<long> run_pdgemm(int m, int n, int k, // matrix sizes
        std::pair<int, int> block_a, // blocks sizes
        std::pair<int, int> block_b,
        std::pair<int, int> block_c,
        int sub_m, int sub_n, int sub_k, // defines submatrices
        char trans_a, char trans_b, // transpose flags
        int p, int q, // processor grid
        double alpha, double beta, // alpha and beta of multiplication
        int rank, int n_rep, // current rank and number of repetitions
        std::string algorithm, // cosma or scalapack
        MPI_Comm comm) {

    // ***********************************
    //   Cblacs context initialization
    // ***********************************
    int myrow, mycol, ctxt;
    char order = 'R';
    // blacs::Cblacs_get(0, 0, &ctxt);
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
    block_a = transpose_if(trans_a, block_a);
    block_b = transpose_if(trans_b, block_b);

    // ********************************************
    //   allocate scalapack buffers for matrices
    // ********************************************
    // get the local number of rows that this rank owns
    int nrows_a = scalapack::numroc_(&am, &block_a.first, &myrow, &rsrc, &p);
    int nrows_b = scalapack::numroc_(&bm, &block_b.first, &myrow, &rsrc, &p);
    int nrows_c = scalapack::numroc_(&cm, &block_c.first, &myrow, &rsrc, &p);

    // get the local number of cols that this rank owns
    int ncols_a = scalapack::numroc_(&an, &block_a.second, &mycol, &csrc, &q);
    int ncols_b = scalapack::numroc_(&bn, &block_b.second, &mycol, &csrc, &q);
    int ncols_c = scalapack::numroc_(&cn, &block_c.second, &mycol, &csrc, &q);

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
    scalapack::descinit_(&desc_a[0], &am, &an, &block_a.first, &block_a.second, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    scalapack::descinit_(&desc_b[0], &bm, &bn, &block_b.first, &block_b.second, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    scalapack::descinit_(&desc_c[0], &cm, &cn, &block_c.first, &block_c.second, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    srand48(rank);
    fillInt(c);

    // vectors to store timings
    std::vector<long> times(n_rep);

    // ***********************************
    //   performing the multiplication
    // ***********************************
    // run COSMA or ScaLAPACK pdgemm n_rep times
    for (int i = 0; i < n_rep; ++i) {
        // clears the profiler
        PC();
        // refill matrices with random data to avoid
        // reusing the cache in subsequent iterations
        fillInt(a);
        fillInt(b);

        long time = 0;

        if (algorithm == "cosma") {
            // ***********************************
            //          run COSMA PDGEMM
            // ***********************************
            // running COSMA wrapper
            MPI_Barrier(comm);
            auto start = std::chrono::steady_clock::now();
            cosma::pdgemm(trans_a, trans_b, m, n, k,
                   alpha, a.data(), ia, ja, &desc_a[0],
                   b.data(), ib, jb, &desc_b[0], beta,
                   c.data(), ic, jc, &desc_c[0]);
            MPI_Barrier(comm);
            auto end = std::chrono::steady_clock::now();
            time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        } else {
            // ***********************************
            //       run ScaLAPACK PDGEMM
            // ***********************************
            // running ScaLAPACK
            MPI_Barrier(comm);
            auto start = std::chrono::steady_clock::now();
            scalapack::pdgemm_(&trans_a, &trans_b, &m, &n, &k,
                   &alpha, a.data(), &ia, &ja, &desc_a[0],
                   b.data(), &ib, &jb, &desc_b[0], &beta,
                   c.data(), &ic, &jc, &desc_c[0]);
            MPI_Barrier(comm);
            auto end = std::chrono::steady_clock::now();
            time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        }
        times[i] = time;
    }

    // exit blacs context
    blacs::Cblacs_gridexit(ctxt);

    // sort cosma timings in increasing order
    std::sort(times.begin(), times.end());

    return times;
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

    // create the context here, so that
    // it doesn't have to be created later
    // (this is not necessary)
    auto ctx = cosma::get_context_instance<double>();
    if (rank == 0) {
        ctx->turn_on_output();
    }

    // **************************************
    //   readout the command line arguments
    // **************************************
    // matrix dimensions
    // dim(A) = mxk, dim(B) = kxn, dim(C) = mxn
    auto m = options::next_int("-m", "--m_dim", "number of rows of A and C.", 1000);
    auto n = options::next_int("-n", "--n_dim", "number of columns of B and C.", 1000);
    auto k = options::next_int("-k", "--k_dim", "number of columns of A and rows of B.", 1000);

    // block sizes
    auto block_a = options::next_int_pair("-ba", "--block_a", "block size for the number of rows of A.", 128);
    auto block_b = options::next_int_pair("-bb", "--block_b", "block size for the number of rows of B.", 128);
    auto block_c = options::next_int_pair("-bc", "--block_c", "block size for the number of rows of C.", 128);

    // indices of submatrices of A, B and C to be multiplied
    // if 1 then full matrices A and B are multiplied
    int submatrix_m = 1;
    int submatrix_n = 1;
    int submatrix_k = 1;

    // processor grid decomposition
    auto p = options::next_int("-p", "--p_row", "number of rows in a processor grid.", 1);
    auto q = options::next_int("-q", "--q_row", "number of columns in a processor grid.", P);

    // alpha and beta of multiplication
    auto alpha = options::next_double("-a", "--alpha", "Alpha parameter in C = alpha*A*B + beta*C", 1.0);
    auto beta = options::next_double("-b", "--beta", "Beta parameter in C = alpha*A*B + beta*C", 0.0);

    // number of repetitions
    auto n_rep = options::next_int("-r", "--n_rep", "number of repetitions", 2);

    // transpose flags
    bool trans_a = options::flag_exists("-ta", "--trans_a");
    bool trans_b = options::flag_exists("-tb", "--trans_b");

    char ta = trans_a ? 'T' : 'N';
    char tb = trans_b ? 'T' : 'N';

    // choose the algorithm
    bool scalapack = options::flag_exists("-s", "--scalapack");
    bool cosma = options::flag_exists("-c", "--cosma");

    std::string algorithm = "cosma";
    if (scalapack == cosma) {
        scalapack = false;
        cosma = true;
    }
    if (scalapack) {
        algorithm = "scalapack";
        cosma = false;
    }
    if (cosma) {
        scalapack = false;
        cosma = true;
        algorithm = "cosma";
    }

    if (p * q != P) {
        std::runtime_error("Number of processors in a grid has to match the number of available ranks.");
    }

    // **************************************
    //    output the problem description
    // **************************************
    if (rank == 0) {
        std::cout << "Running PDGEMM on the following problem size:" << std::endl;
        std::cout << "Matrix sizes: (m, n, k) = (" << m << ", " << n << ", " << k << ")" << std::endl;;
        std::cout << "(alpha, beta) = (" << alpha << ", " << beta << ")" << std::endl;

        std::cout << "Block sizes for A: (" << block_a.first << ", " << block_a.second << ")" << std::endl;
        std::cout << "Block sizes for B: (" << block_b.first << ", " << block_b.second << ")" << std::endl;
        std::cout << "Block sizes for C: (" << block_c.first << ", " << block_c.second << ")" << std::endl;

        std::cout << "Transpose flags (TA, TB) = (" << ta << ", " << tb << ")" << std::endl;
        std::cout << "Processor grid: (prows, pcols) = (" << p << ", " << q << ")" << std::endl;
        std::cout << "Number of repetitions: " << n_rep << std::endl;

        if (scalapack)
            std::cout << "PDGEMM algorithm: ScaLAPACK" << std::endl;
        else
            std::cout << "PDGEMM algorithm: COSMA" << std::endl;
    }

    // *******************************
    //   perform the multiplication
    // ******************************
    std::vector<long> times =
        run_pdgemm(m, n, k,
                   block_a, block_b, block_c,
                   submatrix_m, submatrix_n, submatrix_k,
                   ta, tb,
                   p, q,
                   alpha, beta,
                   rank,
                   n_rep,
                   algorithm,
                   MPI_COMM_WORLD);

    // *****************
    //   output times
    // *****************
    if (rank == 0) {
        std::cout << "PDGEMM TIMES [ms] = ";
        for (auto &time : times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
