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
#include <cosma/pxgemm.h>
#include <cosma/context.hpp>
#include <cosma/profiler.hpp>
#include <cosma/scalapack.hpp>

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
}
}

int main(int argc, char **argv) {
    // ****************************************
    // *       INPUT PARAMETERS BEGIN         *
    // ****************************************
    // *  global coordinates *
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
    int p = 2; // rows
    int q = 4; // cols

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

    // **************************************
    //   setup MPI and command-line parser
    // **************************************
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

    // ************************************
    // *    scalapack processor grid      *
    // ************************************
    char order = 'R';
    int ctxt = cosma::blacs::Csys2blacs_handle(MPI_COMM_WORLD);
    cosma::blacs::Cblacs_gridinit(&ctxt, &order, p, q);

    // ************************************
    // *   scalapack array descriptors    *
    // ************************************
    int info;
    // matrix A
    std::array<int, 9> desca;
    scalapack::descinit_(&desca[0],
                         &ma, &na,
                         &bma, &bna,
                         &src_ma, &src_na,
                         &ctxt,
                         &lld_a,
                         &info);
    if (info != 0) {
        std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // matrix B
    std::array<int, 9> descb;
    scalapack::descinit_(&descb[0],
                         &mb, &nb,
                         &bmb, &bnb,
                         &src_mb, &src_nb,
                         &ctxt,
                         &lld_b,
                         &info);

    if (info != 0) {
        std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // matrix C
    std::array<int, 9> descc;
    scalapack::descinit_(&descc[0],
                         &mc, &nc,
                         &bmc, &bnc,
                         &src_mc, &src_nc,
                         &ctxt,
                         &lld_c,
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
    std::vector<double> c(size_c);

    // initialize the values randomly with 
    // integers stored as doubles
    // for easier debugging
    srand48(rank);
    fill_int(a);
    fill_int(b);
    fill_int(c);

    // ********************************
    // *  perform the multiplication  *
    // ********************************
    MPI_Barrier(MPI_COMM_WORLD);
    pdgemm(&trans_a, &trans_b, &m, &n, &k,
        &alpha, a.data(), &ia, &ja, &desca[0],
        b.data(), &ib, &jb, &descb[0], &beta,
        c.data(), &ic, &jc, &descc[0]);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "COSMA pdgemm finished" << std::endl;

    // exit blacs context
    cosma::blacs::Cblacs_gridexit(ctxt);

    MPI_Finalize();

    return 0;
}
