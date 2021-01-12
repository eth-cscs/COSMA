// from std
#include <algorithm>
#include <array>
#include <iostream>
#include <chrono>
#include <complex>
#include <vector>
#include <cassert>

// from costa
#include <costa/blacs.hpp>
#include <costa/scalapack.hpp>
#include <costa/pxgemr2d/costa_pxgemr2d.hpp>
#include <costa/pxgemr2d/pxgemr2d_params.hpp>
#include <costa/random_generator.hpp>

#include "general.hpp"

// **********************
//   ScaLAPACK routines
// **********************
namespace scalapack {
#ifdef __cplusplus
extern "C" {
#endif
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

    void psgemr2d_(const int *m , const int *n ,
                   const float *a ,
                   const int *ia , const int *ja ,
                   const int *desca ,
                   float *c ,
                   const int *ic , const int *jc ,
                   const int *descc,
                   const int *ctxt);

    void pdgemr2d_(const int *m , const int *n ,
                   const double *a ,
                   const int *ia , const int *ja ,
                   const int *desca ,
                   double *c ,
                   const int *ic , const int *jc ,
                   const int *descc,
                   const int *ctxt);

    void pcgemr2d_(const int *m , const int *n ,
                   const float *a ,
                   const int *ia , const int *ja ,
                   const int *desca ,
                   float *c ,
                   const int *ic , const int *jc ,
                   const int *descc,
                   const int *ctxt);

    void pzgemr2d_(const int *m , const int *n ,
                   const double *a ,
                   const int *ia , const int *ja ,
                   const int *desca ,
                   double *c ,
                   const int *ic , const int *jc ,
                   const int *descc,
                   const int *ctxt);
#ifdef __cplusplus
}
#endif
}

// *************************************
//    templated scalapack pxgemm calls
// *************************************
// this is just for the convenience
template <typename T>
struct scalapack_pxgemr2d {
  static inline void pxgemr2d(
              const int* m, const int* n,
              const T* a,
              const int* ia, const int* ja, const int* desca,
              T* c,
              const int* ic, const int* jc, const int* descc,
              const int *ctxt);
};

template <>
inline void scalapack_pxgemr2d<float>::pxgemr2d(
              const int* m, const int* n,
              const float* a,
              const int* ia, const int* ja, const int* desca,
              float* c,
              const int* ic, const int* jc, const int* descc,
              const int *ctxt) {
    scalapack::psgemr2d_(
                       m, n,
                       a,
                       ia, ja, desca,
                       c,
                       ic, jc, descc,
                       ctxt);
}

template <>
inline void scalapack_pxgemr2d<double>::pxgemr2d(
              const int* m, const int* n,
              const double* a,
              const int* ia, const int* ja, const int* desca,
              double* c,
              const int* ic, const int* jc, const int* descc,
              const int *ctxt) {
    scalapack::pdgemr2d_(
                       m, n,
                       a,
                       ia, ja, desca,
                       c,
                       ic, jc, descc,
                       ctxt);
}

template <>
inline void scalapack_pxgemr2d<std::complex<float>>::pxgemr2d(
              const int* m, const int* n,
              const std::complex<float>* a,
              const int* ia, const int* ja, const int* desca,
              std::complex<float>* c,
              const int* ic, const int* jc, const int* descc,
              const int *ctxt) {
    scalapack::pcgemr2d_(
                       m, n,
                       reinterpret_cast<const float*>(a),
                       ia, ja, desca,
                       reinterpret_cast<float*>(c),
                       ic, jc, descc,
                       ctxt);
}

template <>
inline void scalapack_pxgemr2d<std::complex<double>>::pxgemr2d(
              const int* m, const int* n,
              const std::complex<double>* a,
              const int* ia, const int* ja, const int* desca,
              std::complex<double>* c,
              const int* ic, const int* jc, const int* descc,
              const int *ctxt) {
    scalapack::pzgemr2d_(
                       m, n,
                       reinterpret_cast<const double*>(a),
                       ia, ja, desca,
                       reinterpret_cast<double*>(c),
                       ic, jc, descc,
                       ctxt);
}

// runs costa or scalapack pxgemr2d wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
template <typename T>
bool benchmark_pxgemr2d(costa::pxgemr2d_params<T>& params, MPI_Comm comm, int n_rep,
                    const std::string& algorithm,
                    std::vector<long>& costa_times, std::vector<long>& scalapack_times,
                    bool test_correctness = false, bool exit_blacs = false) {
    assert(algorithm == "both" || algorithm == "costa" || algorithm == "scalapack");
    if (algorithm == "both" || algorithm == "costa") {
        costa_times.resize(n_rep);
    }
    if (algorithm == "both" || algorithm == "scalapack") {
        scalapack_times.resize(n_rep);
    }

    // create the context here, so that
    // it doesn't have to be created later
    // (this is not necessary)
    int rank;
    MPI_Comm_rank(comm, &rank);

#ifdef DEBUG
    if (rank == 0) {
        costa_ctx->turn_on_output();
    }
#endif

    // ************************************
    // *    scalapack processor grid      *
    // ************************************
    int ctxt_a = costa::blacs::Csys2blacs_handle(comm);
    costa::blacs::Cblacs_gridinit(&ctxt_a, &params.order_a, params.p_rows_a, params.p_cols_a);
    int ctxt_c = costa::blacs::Csys2blacs_handle(comm);
    costa::blacs::Cblacs_gridinit(&ctxt_c, &params.order_c, params.p_rows_c, params.p_cols_c);

    int Pa = params.p_rows_a * params.p_cols_a;
    int Pc = params.p_rows_c * params.p_cols_c;

    // union context
    int ctxt_union = Pa > Pc ? ctxt_a : ctxt_c;

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
                         &ctxt_a,
                         &params.lld_a,
                         &info);
    if (rank == 0 && info != 0) {
        std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // matrix C
    std::array<int, 9> descc;
    scalapack::descinit_(&descc[0],
                         &params.mc, &params.nc,
                         &params.bmc, &params.bnc,
                         &params.src_mc, &params.src_nc,
                         &ctxt_c,
                         &params.lld_c,
                         &info);
    if (rank == 0 && info != 0) {
        std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // ************************************
    // *   scalapack memory allocations   *
    // ************************************
    int size_a = costa::scalapack::local_buffer_size(&desca[0]);
    int size_c = costa::scalapack::local_buffer_size(&descc[0]);

    std::vector<T> a;
    std::vector<T> c_costa;
    std::vector<T> c_scalapack;

    try {
        a = std::vector<T>(size_a);
        if (algorithm == "both" || algorithm == "costa") {
            c_costa = std::vector<T>(size_c);
        }
        if (algorithm == "both" || algorithm == "scalapack") {
            c_scalapack = std::vector<T>(size_c);
        }
    } catch (const std::bad_alloc& e) {
        std::cout << "COSTA (pxgemr2d_utils): not enough space to store the initial local matrices. The problem size is too large. Either decrease the problem size or run it on more nodes/ranks." << std::endl;
        costa::blacs::Cblacs_gridexit(ctxt_a);
        costa::blacs::Cblacs_gridexit(ctxt_c);
        int dont_finalize_mpi = 1;
        costa::blacs::Cblacs_exit(dont_finalize_mpi);
        throw;
    } catch (const std::length_error& e) {
        std::cout << "COSTA (pxgemr2d_utils): the initial local size of matrices >= vector::max_size(). Try using std::array or similar in costa/utils/pxgemm_utils.cpp instead of vectors to store the initial matrices." << std::endl;
        costa::blacs::Cblacs_gridexit(ctxt_a);
        costa::blacs::Cblacs_gridexit(ctxt_c);
        int dont_finalize_mpi = 1;
        costa::blacs::Cblacs_exit(dont_finalize_mpi);
        throw;
    } catch (const std::exception& e) {
        std::cout << "COSTA (pxgemr2d_utils): unknown exception, potentially a bug. Please inform us of the test-case." << std::endl;
        costa::blacs::Cblacs_gridexit(ctxt_a);
        costa::blacs::Cblacs_gridexit(ctxt_c);
        int dont_finalize_mpi = 1;
        costa::blacs::Cblacs_exit(dont_finalize_mpi);
        throw;
    }

    for (int i = 0; i < n_rep; ++i) {
        // refill the matrices with random data to avoid
        // reusing the cache in subsequent iterations
        fill_randomly(a);
        if (algorithm == "both") {
            fill_randomly(c_costa);
            // in case beta > 0, this is important in order to get the same results
            c_scalapack = c_costa;
        } else if (algorithm == "costa") {
            fill_randomly(c_costa);
        } else {
            fill_randomly(c_scalapack);
        }

        if (algorithm == "both" || algorithm == "costa") {
            // ***********************************
            //       run COSTA PXGEMR2D
            // ***********************************
            // running COSTA wrapper
            long time = 0;
            MPI_Barrier(comm);
            auto start = std::chrono::steady_clock::now();
            costa::pxgemr2d<T>(
                params.m, params.n,
                a.data(), params.ia, params.ja, &desca[0],
                c_costa.data(), params.ic, params.jc, &descc[0], ctxt_union);
            MPI_Barrier(comm);
            auto end = std::chrono::steady_clock::now();
            time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
            costa_times[i] = time;
        }

        if (algorithm == "both" || algorithm == "scalapack") {
            // ***********************************
            //       run ScaLAPACK PXGEMR2D
            // ***********************************
            // running ScaLAPACK
            long time = 0;
            MPI_Barrier(comm);
            auto start = std::chrono::steady_clock::now();
            scalapack_pxgemr2d<T>::pxgemr2d(
                &params.m, &params.n,
                a.data(), &params.ia, &params.ja, &desca[0],
                c_scalapack.data(), &params.ic, &params.jc, &descc[0], &ctxt_union);
            MPI_Barrier(comm);
            auto end = std::chrono::steady_clock::now();
            time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
            scalapack_times[i] = time;
        }
    }

    if (algorithm == "both" || algorithm == "costa") {
        std::sort(costa_times.rbegin(), costa_times.rend());
    }
    if (algorithm == "both" || algorithm == "scalapack") {
        std::sort(scalapack_times.rbegin(), scalapack_times.rend());
    }

    // if algorithm != both than we don't check the correctness,
    // also if test_correctness flat is set to false
    bool correct = algorithm != "both"
                   || !test_correctness
                   || validate_results(c_costa, c_scalapack);

    // exit blacs context
    costa::blacs::Cblacs_gridexit(ctxt_a);
    costa::blacs::Cblacs_gridexit(ctxt_c);
    if (exit_blacs) {
        int dont_finalize_mpi = 1;
        costa::blacs::Cblacs_exit(dont_finalize_mpi);
    }

    return correct;
}

template <typename T>
bool test_pxgemr2d(costa::pxgemr2d_params<T>& params, MPI_Comm comm,
                 bool test_correctness = true, bool exit_blacs = false) {
    std::vector<long> t1;
    std::vector<long> t2;
    int n_rep = 1;
    return benchmark_pxgemr2d(params, comm, n_rep, "both", t1, t2, true, exit_blacs);
}
