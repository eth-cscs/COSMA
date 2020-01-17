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
#include <cosma/scalapack.hpp>

// from options
#include <options.hpp>

// random number generator
// we cast them to ints, so that we can more easily test them
// but it's not necessary (they are anyway stored as double's)
template <typename T>
void fill_int(T &in) {
    std::generate(in.begin(), in.end(), []() { return (int)(10 * drand48()); });
}

// **********************
//   ScaLAPACK routines
// **********************
namespace scalapack {
extern "C" {
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

    void pdgemm_(const char* trans_a, const char* transb, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);
}
}

// compares two vectors up to eps precision, returns true if they are equal
bool validate_results(std::vector<double>& v1, std::vector<double>& v2) {
    constexpr auto epsilon = std::numeric_limits<double>::epsilon();
    if (v1.size() != v2.size())
        return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

struct pdgemm_params {
    // ****************************************
    // *       INPUT PARAMETERS BEGIN         *
    // ****************************************
    // *  global dimensions  *
    // ***********************
    // matrix A
    int ma = 1280; // rows
    int na = 1280; // cols

    // matrix B
    int mb = 1280; // rows
    int nb = 1280; // cols

    // matrix C
    int mc = 1280; // rows
    int nc = 1280; // cols

    // ***********************
    // *     block sizes     *
    // ***********************
    // matrix A
    int bma = 32; // rows
    int bna = 32; // cols

    // matrix B
    int bmb = 32; // rows
    int bnb = 32; // cols

    // matrix C
    int bmc = 32; // rows
    int bnc = 32; // cols

    // ***********************
    // *   submatrices ij    *
    // ***********************
    // matrix A
    int ia = 1; // rows
    int ja = 545; // cols

    // matrix B
    int ib = 513; // rows
    int jb = 545; // cols

    // matrix C
    int ic = 1; // rows
    int jc = 513; // cols

    // ***********************
    // *    problem size     *
    // ***********************
    int m = 512;
    int n = 32;
    int k = 736;

    // ***********************
    // *   transpose flags   *
    // ***********************
    char trans_a = 'N';
    char trans_b = 'T';

    // ***********************
    // *    scaling flags    *
    // ***********************
    double alpha = 1.0;
    double beta = 1.0;

    // ***********************
    // *    leading dims     *
    // ***********************
    int lld_a = 640;
    int lld_b = 640;
    int lld_c = 640;

    // ***********************
    // *      proc grid      *
    // ***********************
    int p_rows = 2; // rows
    int p_cols = 4; // cols
    int P = p_rows * p_cols;
    char order = 'R';

    // ***********************
    // *      proc srcs      *
    // ***********************
    // matrix A
    int src_ma = 0; // rows
    int src_na = 0; // cols

    // matrix B
    int src_mb = 0; // rows
    int src_nb = 0; // cols

    // matrix C
    int src_mc = 0; // rows
    int src_nc = 0; // cols

    // ****************************************
    // *         INPUT PARAMETERS END         *
    // ****************************************
    pdgemm_params() = default;

    void initialize(int mm, int nn, int kk,
                    int block_a1, int block_a2,
                    int block_b1, int block_b2,
                    int block_c1, int block_c2,
                    int prows, int pcols,
                    char transa, char transb,
                    double a, double b) {
        m = mm;
        n = nn;
        k = kk;

        // global problem size
        // m, n, k are just sizes that we want to multiply
        // starting from (ia-1, ja-1), (ib-1, jb-1) and (ic-1, jc-1)
        // this makes the global problem size m+ia-1, n+jb-1, k+ja-1
        ma = transpose_if(trans_a, m, k);
        na = transpose_if(trans_a, k, m);
        mb = transpose_if(trans_b, k, n);
        nb = transpose_if(trans_b, n, k);
        mc = m;
        nc = n;

        // block sizes
        bma = block_a1;
        bna = block_a2;

        bmb = block_b1;
        bnb = block_b2;

        bmc = block_c1;
        bnc = block_c2;

        // submatrices ij
        ia = 1; ja = 1;
        ib = 1; jb = 1;
        ic = 1; jc = 1;

        // transpose flags
        trans_a = transa; 
        trans_b = transb;

        // scaling parameters
        alpha = a;
        beta = b;

        // leading dims
        int max_n_rows_a = ((ma + bma - 1) / bma) * bma;
        int max_n_rows_b = ((mb + bma - 1) / bmb) * bmb;
        int max_n_rows_c = ((mc + bmc - 1) / bmc) * bmc;

        lld_a = std::max(1, max_n_rows_a);
        lld_b = std::max(1, max_n_rows_b);
        lld_c = std::max(1, max_n_rows_c);

        // proc grid
        order = 'R';
        p_rows = prows;
        p_cols = pcols;
        P = p_rows * p_cols;

        // proc srcs
        src_ma = 0; src_na = 0;
        src_mb = 0; src_nb = 0;
        src_mc = 0; src_nc = 0;
    }

    pdgemm_params(int m, int n, int k,
                  int bm, int bn, int bk,
                  int prows, int pcols,
                  char transa, char transb,
                  double a, double b) {
        // block sizes
        bma = transpose_if(trans_a, bk, bm);
        bna = transpose_if(trans_a, bm, bk);

        bmb = transpose_if(trans_a, bn, bk);
        bnb = transpose_if(trans_a, bk, bn);

        bmc = transpose_if(trans_a, bn, bm);
        bnc = transpose_if(trans_a, bm, bn);

        initialize(m, n, k,
                   bma, bna,
                   bmb, bnb,
                   bmc, bnc,
                   prows, pcols,
                   transa, transb,
                   a, b);
    }


    pdgemm_params(int m, int n, int k,
                  int block_a1, int block_a2,
                  int block_b1, int block_b2,
                  int block_c1, int block_c2,
                  int prows, int pcols,
                  char transa, char transb,
                  double a, double b) {
        initialize(m, n, k,
                   block_a1, block_a2,
                   block_b1, block_b2,
                   block_c1, block_c2,
                   prows, pcols,
                   transa, transb,
                   a, b);
    }

    pdgemm_params(
        // global sizes
        int ma, int na, // matrix A
        int mb, int nb, // matrix B
        int mc, int nc, // matrix C

        // block sizes
        int bma, int bna, // matrix A
        int bmb, int bnb, // matrix B
        int bmc, int bnc, // matrix C

        // submatrices ij
        int ia, int ja, // matrix A
        int ib, int jb, // matrix B
        int ic, int jc, // matrix C

        // problem size
        int m, int n, int k,

        // transpose flags
        char trans_a, char trans_b,

        // scaling flags
        double alpha, double beta,

        // leading dimensions
        int lld_a, int lld_b, int lld_c,

        // processor grid
        int p_rows, int p_cols,
        char order,

        // processor srcs
        int src_ma, int src_na, // matrix A
        int src_mb, int src_nb, // matrix B
        int src_mc, int src_nc // matrix C
    ) :
        ma(ma), na(na),
        mb(mb), nb(nb),
        mc(mc), nc(nc),

        bma(bma), bna(bna),
        bmb(bmb), bnb(bnb),
        bmc(bmc), bnc(bnc),

        ia(ia), ja(ja),
        ib(ib), jb(jb),
        ic(ic), jc(jc),

        m(m), n(n), k(k),

        trans_a(trans_a), trans_b(trans_b),

        alpha(alpha), beta(beta),

        lld_a(lld_a), lld_b(lld_b), lld_c(lld_c),

        order(order),
        p_rows(p_rows), p_cols(p_cols),
        P(p_rows * p_cols),

        src_ma(src_ma), src_na(src_na),
        src_mb(src_mb), src_nb(src_nb),
        src_mc(src_mc), src_nc(src_nc)
    {}

    int transpose_if(char transpose_flag, int row, int col) {
        return transpose_flag == 'N' ? row : col;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const pdgemm_params &obj) {
        os << "=============================" << std::endl;
        os << "      GLOBAL MAT. SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "A = " << obj.ma << " x " << obj.na << std::endl;
        os << "B = " << obj.mb << " x " << obj.nb << std::endl;
        os << "C = " << obj.mc << " x " << obj.nc << std::endl;
        os << "=============================" << std::endl;
        os << "        SUBMATRICES" << std::endl;
        os << "=============================" << std::endl;
        os << "(ia, ja) = (" << obj.ia << ", " << obj.ja << ")" << std::endl;
        os << "(ib, jb) = (" << obj.ib << ", " << obj.jb << ")" << std::endl;
        os << "(ic, jc) = (" << obj.ic << ", " << obj.jc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "      SUBMATRIX SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "m = " << obj.m << std::endl;
        os << "n = " << obj.n << std::endl;
        os << "k = " << obj.k << std::endl;
        os << "=============================" << std::endl;
        os << "      ADDITIONAL OPTIONS" << std::endl;
        os << "=============================" << std::endl;
        os << "alpha = " << obj.alpha << std::endl;
        os << "beta = " << obj.beta << std::endl;
        os << "trans_a = " << obj.trans_a << std::endl;
        os << "trans_b = " << obj.trans_b << std::endl;
        os << "=============================" << std::endl;
        os << "         PROC GRID" << std::endl;
        os << "=============================" << std::endl;
        os << "grid = " << obj.p_rows << " x " << obj.p_cols << std::endl;
        os << "grid order = " << obj.order << std::endl;
        os << "=============================" << std::endl;
        os << "         PROC SRCS" << std::endl;
        os << "=============================" << std::endl;
        os << "P_SRC(A) = (" << obj.src_ma << ", " << obj.src_na << std::endl;
        os << "P_SRC(B) = (" << obj.src_mb << ", " << obj.src_nb << std::endl;
        os << "P_SRC(C) = (" << obj.src_mc << ", " << obj.src_nc << std::endl;
        os << "=============================" << std::endl;
        os << "          BLOCK SIZES" << std::endl;
        os << "=============================" << std::endl;
        os << "Blocks(A) = (" << obj.bma << ", " << obj.bna << ")" << std::endl;
        os << "Blocks(B) = (" << obj.bmb << ", " << obj.bnb << ")" << std::endl;
        os << "Blocks(C) = (" << obj.bmc << ", " << obj.bnc << ")" << std::endl;
        os << "=============================" << std::endl;
        os << "          LEADING DIMS" << std::endl;
        os << "=============================" << std::endl;
        os << "lld_a = " << obj.lld_a << std::endl;
        os << "lld_b = " << obj.lld_b << std::endl;
        os << "lld_c = " << obj.lld_c << std::endl;
        os << "=============================" << std::endl;
        return os;
    }
};

// runs cosma or scalapack pdgemm wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
bool test_pdgemm(pdgemm_params& params, MPI_Comm comm) {
    // create the context here, so that
    // it doesn't have to be created later
    // (this is not necessary)
    int rank;
    MPI_Comm_rank(comm, &rank);
    auto ctx = cosma::get_context_instance<double>();
    if (rank == 0) {
        ctx->turn_on_output();
    }

    // ************************************
    // *    scalapack processor grid      *
    // ************************************
    int ctxt = cosma::blacs::Csys2blacs_handle(comm);
    cosma::blacs::Cblacs_gridinit(&ctxt, &params.order, params.p_rows, params.p_cols);

    // ************************************
    // *   scalapack array descriptors    *
    // ************************************
    int info;
    // matrix A
    std::array<int, 9> desca;
    scalapack::descinit_(&desca[0],
                         &params.ma, &params.na,
                         &params.bma, &params.bna,
                         &params.src_ma, &params.src_na,
                         &ctxt,
                         &params.lld_a,
                         &info);
    if (info != 0) {
        std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // matrix B
    std::array<int, 9> descb;
    scalapack::descinit_(&descb[0],
                         &params.mb, &params.nb,
                         &params.bmb, &params.bnb,
                         &params.src_mb, &params.src_nb,
                         &ctxt,
                         &params.lld_b,
                         &info);

    if (info != 0) {
        std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // matrix C
    std::array<int, 9> descc;
    scalapack::descinit_(&descc[0],
                         &params.mc, &params.nc,
                         &params.bmc, &params.bnc,
                         &params.src_mc, &params.src_nc,
                         &ctxt,
                         &params.lld_c,
                         &info);
    if (info != 0) {
        std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // ************************************
    // *   scalapack memory allocations   *
    // ************************************
    int size_a = cosma::scalapack::local_buffer_size(&desca[0]);
    int size_b = cosma::scalapack::local_buffer_size(&descb[0]);
    int size_c = cosma::scalapack::local_buffer_size(&descc[0]);

    std::vector<double> a(size_a);
    std::vector<double> b(size_b);
    std::vector<double> c_cosma(size_c);
    std::vector<double> c_scalapack(size_c);

    // fill the matrices with random data
    srand48(rank);

    fill_int(a);
    fill_int(b);
    fill_int(c_cosma);
    // in case beta > 0, this is important in order to get the same results
    c_scalapack = c_cosma;


    // ***********************************
    //       run COSMA PDGEMM
    // ***********************************
    // running COSMA wrapper
    cosma::pgemm<double>(
           params.trans_a, params.trans_b, 
           params.m, params.n, params.k,
           params.alpha, a.data(), params.ia, params.ja, &desca[0],
           b.data(), params.ib, params.jb, &descb[0], params.beta,
           c_cosma.data(), params.ic, params.jc, &descc[0]);

    // ***********************************
    //       run ScaLAPACK PDGEMM
    // ***********************************
    // running ScaLAPACK
    scalapack::pdgemm_(
           &params.trans_a, &params.trans_b, 
           &params.m, &params.n, &params.k,
           &params.alpha, a.data(), &params.ia, &params.ja, &desca[0],
           b.data(), &params.ib, &params.jb, &descb[0], &params.beta,
           c_scalapack.data(), &params.ic, &params.jc, &descc[0]);

#ifdef DEBUG
    if (myrow == 0 && mycol == 0) {
        std::cout << "c(cosma) = ";
        for (int i = 0; i < c_cosma.size(); ++i) {
            std::cout << c_cosma[i] << ", ";
        }
        std::cout << std::endl;
        std::cout << "c(scalapack) = ";
        for (int i = 0; i < c_scalapack.size(); ++i) {
            std::cout << c_scalapack[i] << ", ";
        }
        std::cout << std::endl;
    }
#endif

    // exit blacs context
    cosma::blacs::Cblacs_gridexit(ctxt);

    return validate_results(c_cosma, c_scalapack);
}
