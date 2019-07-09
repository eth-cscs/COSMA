#ifdef COSMA_WITH_SCALAPACK
// from std
#include <cassert>
#include <complex>
#include <mpi.h>
// from cosma
#include <cosma/blacs.hpp>
#include <cosma/multiply.hpp>
#include <cosma/pgemm.hpp>
#include <cosma/scalapack.hpp>
// from grid2grid
#include <transform.hpp>

namespace cosma {
using zdouble_t = std::complex<double>;
using zfloat_t = std::complex<float>;

// alpha ignored at the moment
template <typename T>
void pgemm(const char trans_a, const char trans_b, const int m, const int n, const int k,
           const T alpha, const T* a, const int ia, const int ja, const int* desca,
           const T* b, const int ib, const int jb, const int* descb, const T beta,
           T* c, const int ic, const int jc, const int* descc) {
    // blas context
    int ctxt = scalapack::get_context(desca, descb, descc);

    // communicator size and rank
    int rank, P;
    blacs::Cblacs_pinfo(&rank, &P);

    // scalapack rank grid decomposition
    int procrows, proccols;
    int myrow, mycol;
    blacs::Cblacs_gridinfo(ctxt, &procrows, &proccols, &myrow, &mycol);

    // get MPI communicator
    MPI_Comm comm = blacs::Cblacs2sys_handle(ctxt);

    // block sizes
    scalapack::block_sizes b_dims(desca, descb, descc);

    // global matrix sizes
    scalapack::global_matrix_sizes m_dims(desca, descb, descc);

    // rank sources (rank coordinates that own first row and column of a matrix)
    scalapack::rank_src rank_src_a(desca);
    scalapack::rank_src rank_src_b(descb);
    scalapack::rank_src rank_src_c(descc);

    // leading dimensions
    int lld_a = scalapack::leading_dimension(desca);
    int lld_b = scalapack::leading_dimension(descb);
    int lld_c = scalapack::leading_dimension(descc);

    // transpose flags
    bool trans_a_flag = trans_a == 'N';
    bool trans_b_flag = trans_b == 'N';
    bool trans_c_flag = 'N';

    // check whether rank grid is row-major or col-major
    auto ordering = scalapack::rank_ordering(ctxt, P);

    // find an optimal strategy for this problem
    Strategy strategy(m, n, k, P);

    // create COSMA matrices
    CosmaMatrix<T> A('A', strategy, rank);
    CosmaMatrix<T> B('B', strategy, rank);
    CosmaMatrix<T> C('C', strategy, rank);

    // get abstract layout descriptions for COSMA layout
    auto cosma_layout_a = A.get_grid_layout();
    auto cosma_layout_b = B.get_grid_layout();
    auto cosma_layout_c = C.get_grid_layout();

    // get abstract layout descriptions for ScaLAPACK layout
    auto scalapack_layout_a = grid2grid::get_scalapack_grid<T>(
        lld_a,
        {m_dims.m, m_dims.k},
        {ia, ja},
        {m, k},
        {b_dims.m, b_dims.k},
        {procrows, proccols},
        ordering,
        trans_a_flag,
        {rank_src_a.row_src, rank_src_a.col_src},
        const_cast<T*>(a), rank
    );

    auto scalapack_layout_b = grid2grid::get_scalapack_grid<T>(
        lld_b,
        {m_dims.k, m_dims.n},
        {ib, jb},
        {k, n},
        {b_dims.k, b_dims.n},
        {procrows, proccols},
        ordering,
        trans_b_flag,
        {rank_src_b.row_src, rank_src_b.col_src},
        const_cast<T*>(b), rank
    );

    auto scalapack_layout_c = grid2grid::get_scalapack_grid<T>(
        lld_c,
        {m_dims.m, m_dims.n},
        {ic, jc},
        {m, n},
        {b_dims.m, b_dims.n},
        {procrows, proccols},
        ordering,
        trans_c_flag,
        {rank_src_c.row_src, rank_src_c.col_src},
        c, rank
    );

    // transform A and B from scalapack to cosma layout
    grid2grid::transform<T>(scalapack_layout_a, cosma_layout_a, comm);
    grid2grid::transform<T>(scalapack_layout_b, cosma_layout_b, comm);

    // perform cosma multiplication
    auto ctx = cosma::make_context();
    multiply<T>(ctx, A, B, C, strategy, comm, beta);

    // transform the result from cosma back to scalapack
    grid2grid::transform<T>(cosma_layout_c, scalapack_layout_c, comm);
}

// explicit instantiation for pgemm
template void pgemm<double>(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const double alpha, const double* a, const int ia, const int ja, const int* desca,
    const double* b, const int ib, const int jb, const int* descb, const double beta,
    double* c, const int ic, const int jc, const int* descc);

template void pgemm<float>(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const float alpha, const float* a, const int ia, const int ja, const int* desca,
    const float* b, const int ib, const int jb, const int* descb, const float beta,
    float* c, const int ic, const int jc, const int* descc);

template void pgemm<zdouble_t>(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const zdouble_t alpha, const zdouble_t* a, const int ia, const int ja, const int* desca,
    const zdouble_t* b, const int ib, const int jb, const int* descb, const zdouble_t beta,
    zdouble_t* c, const int ic, const int jc, const int* descc);

template void pgemm<zfloat_t>(const char trans_a, const char trans_b, 
    const int m, const int n, const int k,
    const zfloat_t alpha, const zfloat_t* a, const int ia, const int ja, const int* desca,
    const zfloat_t* b, const int ib, const int jb, const int* descb, const zfloat_t beta,
    zfloat_t* c, const int ic, const int jc, const int* descc);

}
#endif
