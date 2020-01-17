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
#include <cosma/cosma_pxgemm.hpp>
#include <cosma/scalapack.hpp>
#include <cosma/pxgemm_params.hpp>

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

// runs cosma or scalapack pdgemm wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
bool test_pdgemm(pxgemm_params<double>& params, MPI_Comm comm) {
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
    cosma::pxgemm<double>(
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
