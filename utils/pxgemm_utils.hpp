// from std
#include <algorithm>
#include <array>
#include <iostream>
#include <chrono>
#include <complex>
#include <vector>

// from cosma
#include <cosma/blacs.hpp>
#include <cosma/scalapack.hpp>
#include <cosma/cosma_pxgemm.hpp>
#include <cosma/pxgemm_params.hpp>
#include <cosma/random_generator.hpp>
#include <cosma/context.hpp>
#include <cosma/profiler.hpp>

// random number generator
// we cast them to ints, so that we can more easily test them
// but it's not necessary (they are anyway stored as double's)
template <typename T>
void fill_randomly(std::vector<T> &in) {
    std::generate(in.begin(), in.end(), []() { return cosma::random_generator<T>::sample();});
}

// **********************
//   ScaLAPACK routines
// **********************
namespace scalapack {
#ifdef __cplusplus
extern "C" {
#endif
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

    void psgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
            const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
            float* c, const int* ic, const int* jc, const int* descc);

    void pdgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);
#ifdef __cplusplus
}
#endif
}

// *************************************
//    templated scalapack pxgemm calls
// *************************************
// this is just for the convenience
template <typename T>
struct scalapack_pxgemm {
  static inline void pxgemm(
              const char* trans_a, const char* trans_b, 
              const int* m, const int* n, const int* k,
              const T* alpha, const T* a, 
              const int* ia, const int* ja, const int* desca,
              const T* b, 
              const int* ib, const int* jb, const int* descb, 
              const T* beta, T* c, 
              const int* ic, const int* jc, const int* descc);
};

template <>
inline void scalapack_pxgemm<float>::pxgemm(
              const char* trans_a, const char* trans_b, 
              const int* m, const int* n, const int* k,
              const float* alpha, const float* a, 
              const int* ia, const int* ja, const int* desca,
              const float* b, 
              const int* ib, const int* jb, const int* descb, 
              const float* beta, float* c, 
              const int* ic, const int* jc, const int* descc) {
    scalapack::psgemm_(trans_a, trans_b,
                       m, n, k,
                       alpha, a,
                       ia, ja, desca,
                       b,
                       ib, jb, descb,
                       beta, c,
                       ic, jc, descc);
}

template <>
inline void scalapack_pxgemm<double>::pxgemm(
              const char* trans_a, const char* trans_b,
              const int* m, const int* n, const int* k,
              const double* alpha, const double* a, 
              const int* ia, const int* ja, const int* desca,
              const double* b, 
              const int* ib, const int* jb, const int* descb,
              const double* beta, double* c, 
              const int* ic, const int* jc, const int* descc) {
    scalapack::pdgemm_(trans_a, trans_b,
                       m, n, k,
                       alpha, a,
                       ia, ja, desca,
                       b,
                       ib, jb, descb,
                       beta, c,
                       ic, jc, descc);
}

// compares two vectors up to eps precision, returns true if they are equal
template <typename T>
bool validate_results(std::vector<T>& v1, std::vector<T>& v2, double epsilon=1e-8) {
    if (v1.size() != v2.size())
        return false;
    if (v1.size() == 0)
        return true;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > epsilon) {
            std::cout << "epsilon = " << epsilon << ", v1 = " << v1[i] << ", which is != " << v2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// runs cosma or scalapack pdgemm wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
template <typename T>
bool test_pdgemm(cosma::pxgemm_params<T>& params, MPI_Comm comm, double epsilon=1e-8) {
    // create the context here, so that
    // it doesn't have to be created later
    // (this is not necessary)
    int rank;
    MPI_Comm_rank(comm, &rank);
    auto cosma_ctx = cosma::get_context_instance<T>();
#ifdef DEBUG
    if (rank == 0) {
        cosma_ctx->turn_on_output();
    }
#endif

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
    if (rank == 0 && info != 0) {
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

    if (rank == 0 && info != 0) {
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
    if (rank == 0 && info != 0) {
        std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // ************************************
    // *   scalapack memory allocations   *
    // ************************************
    int size_a = cosma::scalapack::local_buffer_size(&desca[0]);
    int size_b = cosma::scalapack::local_buffer_size(&descb[0]);
    int size_c = cosma::scalapack::local_buffer_size(&descc[0]);

    std::vector<T> a(size_a);
    std::vector<T> b(size_b);
    std::vector<T> c_cosma(size_c);
    std::vector<T> c_scalapack(size_c);

    // fill the matrices with random data
    fill_randomly(a);
    fill_randomly(b);
    fill_randomly(c_cosma);
    // in case beta > 0, this is important in order to get the same results
    c_scalapack = c_cosma;

    // ***********************************
    //       run ScaLAPACK PDGEMM
    // ***********************************
    // running ScaLAPACK
    scalapack_pxgemm<T>::pxgemm(
           &params.trans_a, &params.trans_b,
           &params.m, &params.n, &params.k,
           &params.alpha, a.data(), &params.ia, &params.ja, &desca[0],
           b.data(), &params.ib, &params.jb, &descb[0], &params.beta,
           c_scalapack.data(), &params.ic, &params.jc, &descc[0]);

    // ***********************************
    //       run COSMA PDGEMM
    // ***********************************
    // running COSMA wrapper
    cosma::pxgemm<T>(
           params.trans_a, params.trans_b, 
           params.m, params.n, params.k,
           params.alpha, a.data(), params.ia, params.ja, &desca[0],
           b.data(), params.ib, params.jb, &descb[0], params.beta,
           c_cosma.data(), params.ic, params.jc, &descc[0]);

#ifdef DEBUG
    if (rank == 0) {
        std::cout << "c(cosma) = ";
        for (int i = 0; i < std::min(c_cosma.size(), (std::size_t) 10); ++i) {
            std::cout << c_cosma[i] << ", ";
        }
        std::cout << std::endl;
        std::cout << "c(scalapack) = ";
        for (int i = 0; i < std::min(c_scalapack.size(), (std::size_t) 10); ++i) {
            std::cout << c_scalapack[i] << ", ";
        }
        std::cout << std::endl;
    }
#endif

    // exit blacs context
    cosma::blacs::Cblacs_gridexit(ctxt);

    return validate_results(c_cosma, c_scalapack, epsilon);
}

// runs cosma or scalapack pdgemm wrapper for n_rep times and returns
// a vector of timings (in milliseconds) of size n_rep
template <typename T>
void benchmark_pxgemm(cosma::pxgemm_params<T>& params, MPI_Comm comm, int n_rep,
                    std::vector<long>& cosma_times, std::vector<long>& scalapack_times) {
    cosma_times.resize(n_rep);
    scalapack_times.resize(n_rep);

    // create the context here, so that
    // it doesn't have to be created later
    // (this is not necessary)
    int rank;
    MPI_Comm_rank(comm, &rank);
    auto cosma_ctx = cosma::get_context_instance<T>();
#ifdef DEBUG
    if (rank == 0) {
        cosma_ctx->turn_on_output();
    }
#endif

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
    if (rank == 0 && info != 0) {
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

    if (rank == 0 && info != 0) {
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
    if (rank == 0 && info != 0) {
        std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
    }

    // ************************************
    // *   scalapack memory allocations   *
    // ************************************
    int size_a = cosma::scalapack::local_buffer_size(&desca[0]);
    int size_b = cosma::scalapack::local_buffer_size(&descb[0]);
    int size_c = cosma::scalapack::local_buffer_size(&descc[0]);

    std::vector<T> a(size_a);
    std::vector<T> b(size_b);
    std::vector<T> c_cosma(size_c);
    std::vector<T> c_scalapack(size_c);

    for (int i = 0; i < n_rep; ++i) {
        // clears the profiler
        PC();
        // refill the matrices with random data to avoid
        // reusing the cache in subsequent iterations
        fill_randomly(a);
        fill_randomly(b);
        fill_randomly(c_cosma);
        // in case beta > 0, this is important in order to get the same results
        c_scalapack = c_cosma;

        // ***********************************
        //       run COSMA PDGEMM
        // ***********************************
        // running COSMA wrapper
        long time = 0;
        MPI_Barrier(comm);
        auto start = std::chrono::steady_clock::now();
        cosma::pxgemm<T>(
               params.trans_a, params.trans_b, 
               params.m, params.n, params.k,
               params.alpha, a.data(), params.ia, params.ja, &desca[0],
               b.data(), params.ib, params.jb, &descb[0], params.beta,
               c_cosma.data(), params.ic, params.jc, &descc[0]);
        MPI_Barrier(comm);
        auto end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        cosma_times[i] = time;

        // ***********************************
        //       run ScaLAPACK PDGEMM
        // ***********************************
        // running ScaLAPACK
        time = 0;
        MPI_Barrier(comm);
        start = std::chrono::steady_clock::now();
        scalapack_pxgemm<T>::pxgemm(
               &params.trans_a, &params.trans_b,
               &params.m, &params.n, &params.k,
               &params.alpha, a.data(), &params.ia, &params.ja, &desca[0],
               b.data(), &params.ib, &params.jb, &descb[0], &params.beta,
               c_scalapack.data(), &params.ic, &params.jc, &descc[0]);
        MPI_Barrier(comm);
        end = std::chrono::steady_clock::now();
        time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        scalapack_times[i] = time;
    }

    std::sort(cosma_times.begin(), cosma_times.end());
    std::sort(scalapack_times.begin(), scalapack_times.end());

    // exit blacs context
    cosma::blacs::Cblacs_gridexit(ctxt);
}
