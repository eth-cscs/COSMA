#include <cosma/blacs.hpp>
#include <cosma/multiply.hpp>
#include <cosma/pgemm.hpp>
#include <cosma/profiler.hpp>
#include <cosma/scalapack.hpp>

#include <grid2grid/transform.hpp>

#include <cassert>
#include <complex>
#include <mpi.h>

namespace cosma {
template <typename T>
void pgemm(const char trans_a,
           const char trans_b,
           const int m,
           const int n,
           const int k,
           const T alpha,
           const T *a,
           const int ia,
           const int ja,
           const int *desca,
           const T *b,
           const int ib,
           const int jb,
           const int *descb,
           const T beta,
           T *c,
           const int ic,
           const int jc,
           const int *descc) {
    // blas context
    int ctxt = scalapack::get_grid_context(desca, descb, descc);

    // scalapack rank grid decomposition
    int procrows, proccols;
    int myrow, mycol;
    blacs::Cblacs_gridinfo(ctxt, &procrows, &proccols, &myrow, &mycol);

    // get MPI communicator
    MPI_Comm comm = scalapack::get_communicator(ctxt);

    // communicator size and rank
    int rank, P;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    // block sizes
    scalapack::block_size b_dim_a(desca);
    scalapack::block_size b_dim_b(descb);
    scalapack::block_size b_dim_c(descc);

    // global matrix sizes
    scalapack::global_matrix_size mat_dim_a(desca);
    scalapack::global_matrix_size mat_dim_b(descb);
    scalapack::global_matrix_size mat_dim_c(descc);

    // sumatrix size to multiply
    int a_subm = trans_a == 'N' ? m : k;
    int a_subn = trans_a == 'N' ? k : m;

    int b_subm = trans_b == 'N' ? k : n;
    int b_subn = trans_b == 'N' ? n : k;

    int c_subm = m;
    int c_subn = n;

    // rank sources (rank coordinates that own first row and column of a matrix)
    scalapack::rank_src rank_src_a(desca);
    scalapack::rank_src rank_src_b(descb);
    scalapack::rank_src rank_src_c(descc);

    // leading dimensions
    int lld_a = scalapack::leading_dimension(desca);
    int lld_b = scalapack::leading_dimension(descb);
    int lld_c = scalapack::leading_dimension(descc);

    // check whether rank grid is row-major or col-major
    auto ordering = scalapack::rank_ordering(ctxt, P);

    PE(strategy);
    // find an optimal strategy for this problem
    Strategy strategy(m, n, k, P);
    // if (rank == 0) {
    //     std::cout << strategy << std::endl;
    // }
    PL();

    // create COSMA matrices
    CosmaMatrix<T> A('A', strategy, rank);
    CosmaMatrix<T> B('B', strategy, rank);
    CosmaMatrix<T> C('C', strategy, rank);

    // not necessary since multiply will invoke it
    // and matrix_pointer in matrix class is not outdated
    // initialize COSMA matrices
    // A.initialize();
    // B.initialize();
    // C.initialize();

    PE(transformation_initialization);
    // get abstract layout descriptions for COSMA layout
    auto cosma_layout_a = A.get_grid_layout();
    auto cosma_layout_b = B.get_grid_layout();

    // get abstract layout descriptions for ScaLAPACK layout
    auto scalapack_layout_a = grid2grid::get_scalapack_grid<T>(
        lld_a,
        {mat_dim_a.rows, mat_dim_a.cols},
        {ia, ja},
        {a_subm, a_subn},
        {b_dim_a.rows, b_dim_a.cols},
        {procrows, proccols},
        ordering,
        trans_a,
        {rank_src_a.row_src, rank_src_a.col_src},
        a,
        rank);

    auto scalapack_layout_b = grid2grid::get_scalapack_grid<T>(
        lld_b,
        {mat_dim_b.rows, mat_dim_b.cols},
        {ib, jb},
        {b_subm, b_subn},
        {b_dim_b.rows, b_dim_b.cols},
        {procrows, proccols},
        ordering,
        trans_b,
        {rank_src_b.row_src, rank_src_b.col_src},
        b,
        rank);

    auto scalapack_layout_c = grid2grid::get_scalapack_grid<T>(
        lld_c,
        {mat_dim_c.rows, mat_dim_c.cols},
        {ic, jc},
        {c_subm, c_subn},
        {b_dim_c.rows, b_dim_c.cols},
        {procrows, proccols},
        ordering,
        'N',
        {rank_src_c.row_src, rank_src_c.col_src},
        c,
        rank);
    PL();

#ifdef DEBUG
    std::cout << "Transforming the input matrices A and B from Scalapack -> COSMA" << std::endl;
#endif
    PE(transformation_cosma2scalapack);
    // transform A and B from scalapack to cosma layout
    grid2grid::transform<T>(scalapack_layout_a, cosma_layout_a, comm);
    grid2grid::transform<T>(scalapack_layout_b, cosma_layout_b, comm);

    // transform C from scalapack to cosma only if beta > 0
    if (std::abs(beta) > 0) {
        auto cosma_layout_c = C.get_grid_layout();
        grid2grid::transform<T>(scalapack_layout_c, cosma_layout_c, comm);
    }
    PL();

    // perform cosma multiplication
#ifdef DEBUG
    std::cout << "COSMA multiply" << std::endl;
#endif
    multiply<T>(A, B, C, strategy, comm, alpha, beta);

    auto cosma_layout_c = C.get_grid_layout();
#ifdef DEBUG
    std::cout << "Transforming the result C back from COSMA to ScaLAPACK" << std::endl;
#endif
    PE(transformation_scalapack2cosma);
    // transform the result from cosma back to scalapack
    grid2grid::transform<T>(cosma_layout_c, scalapack_layout_c, comm);
    PL();
}

// explicit instantiation for pgemm
template void pgemm<double>(const char trans_a,
                            const char trans_b,
                            const int m,
                            const int n,
                            const int k,
                            const double alpha,
                            const double *a,
                            const int ia,
                            const int ja,
                            const int *desca,
                            const double *b,
                            const int ib,
                            const int jb,
                            const int *descb,
                            const double beta,
                            double *c,
                            const int ic,
                            const int jc,
                            const int *descc);

template void pgemm<float>(const char trans_a,
                           const char trans_b,
                           const int m,
                           const int n,
                           const int k,
                           const float alpha,
                           const float *a,
                           const int ia,
                           const int ja,
                           const int *desca,
                           const float *b,
                           const int ib,
                           const int jb,
                           const int *descb,
                           const float beta,
                           float *c,
                           const int ic,
                           const int jc,
                           const int *descc);

template void pgemm<zdouble_t>(const char trans_a,
                               const char trans_b,
                               const int m,
                               const int n,
                               const int k,
                               const zdouble_t alpha,
                               const zdouble_t *a,
                               const int ia,
                               const int ja,
                               const int *desca,
                               const zdouble_t *b,
                               const int ib,
                               const int jb,
                               const int *descb,
                               const zdouble_t beta,
                               zdouble_t *c,
                               const int ic,
                               const int jc,
                               const int *descc);

template void pgemm<zfloat_t>(const char trans_a,
                              const char trans_b,
                              const int m,
                              const int n,
                              const int k,
                              const zfloat_t alpha,
                              const zfloat_t *a,
                              const int ia,
                              const int ja,
                              const int *desca,
                              const zfloat_t *b,
                              const int ib,
                              const int jb,
                              const int *descb,
                              const zfloat_t beta,
                              zfloat_t *c,
                              const int ic,
                              const int jc,
                              const int *descc);

// Reimplement ScaLAPACK signatures functions
void pdgemm(const char trans_a,
            const char trans_b,
            const int m,
            const int n,
            const int k,
            const double alpha,
            const double *a,
            const int ia,
            const int ja,
            const int *desca,
            const double *b,
            const int ib,
            const int jb,
            const int *descb,
            const double beta,
            double *c,
            const int ic,
            const int jc,
            const int *descc) {

    pgemm<double>(trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  alpha,
                  a,
                  ia,
                  ja,
                  desca,
                  b,
                  ib,
                  jb,
                  descb,
                  beta,
                  c,
                  ic,
                  jc,
                  descc);
}

void psgemm(const char trans_a,
            const char trans_b,
            const int m,
            const int n,
            const int k,
            const float alpha,
            const float *a,
            const int ia,
            const int ja,
            const int *desca,
            const float *b,
            const int ib,
            const int jb,
            const int *descb,
            const float beta,
            float *c,
            const int ic,
            const int jc,
            const int *descc) {
    pgemm<float>(trans_a,
                 trans_b,
                 m,
                 n,
                 k,
                 alpha,
                 a,
                 ia,
                 ja,
                 desca,
                 b,
                 ib,
                 jb,
                 descb,
                 beta,
                 c,
                 ic,
                 jc,
                 descc);
}

void pcgemm(const char trans_a,
            const char trans_b,
            const int m,
            const int n,
            const int k,
            const zfloat_t alpha,
            const zfloat_t *a,
            const int ia,
            const int ja,
            const int *desca,
            const zfloat_t *b,
            const int ib,
            const int jb,
            const int *descb,
            const zfloat_t beta,
            zfloat_t *c,
            const int ic,
            const int jc,
            const int *descc) {

    pgemm<zfloat_t>(trans_a,
                    trans_b,
                    m,
                    n,
                    k,
                    alpha,
                    a,
                    ia,
                    ja,
                    desca,
                    b,
                    ib,
                    jb,
                    descb,
                    beta,
                    c,
                    ic,
                    jc,
                    descc);
}

void pzgemm(const char trans_a,
            const char trans_b,
            const int m,
            const int n,
            const int k,
            const zdouble_t alpha,
            const zdouble_t *a,
            const int ia,
            const int ja,
            const int *desca,
            const zdouble_t *b,
            const int ib,
            const int jb,
            const int *descb,
            const zdouble_t beta,
            zdouble_t *c,
            const int ic,
            const int jc,
            const int *descc) {

    pgemm<zdouble_t>(trans_a,
                     trans_b,
                     m,
                     n,
                     k,
                     alpha,
                     a,
                     ia,
                     ja,
                     desca,
                     b,
                     ib,
                     jb,
                     descb,
                     beta,
                     c,
                     ic,
                     jc,
                     descc);
}
} // namespace cosma
