#include <cosma/blacs.hpp>
#include <cosma/multiply.hpp>
#include <cosma/pxgemm.hpp>
#include <cosma/profiler.hpp>
#include <cosma/scalapack.hpp>
#include <grid2grid/ranks_reordering.hpp>

#include <grid2grid/transformer.hpp>

#include <cassert>
#include <mpi.h>

namespace cosma {
template <typename T>
void pxgemm(const char transa,
           const char transb,
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
    char trans_a = std::toupper(transa);
    char trans_b = std::toupper(transb);

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
    // strategy.overlap_comm_and_comp = true;
    PL();
#ifdef DEBUG
    if (rank == 0) {
        std::cout << strategy << std::endl;
    }
#endif

    // create COSMA mappers
    Mapper mapper_a('A', strategy, rank);
    Mapper mapper_b('B', strategy, rank);
    Mapper mapper_c('C', strategy, rank);

    auto cosma_grid_a = mapper_a.get_layout_grid();
    auto cosma_grid_b = mapper_b.get_layout_grid();
    auto cosma_grid_c = mapper_c.get_layout_grid();

    // if (rank == 0) {
    //     std::cout << "COSMA grid for A before reordering: " << cosma_grid_a << std::endl;
    // }

    PE(transform_init);
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

    PE(transform_reordering_matching);
    // total communication volume for transformation of layouts
    auto comm_vol = grid2grid::communication_volume(scalapack_layout_a.grid, cosma_grid_a);
    comm_vol += grid2grid::communication_volume(scalapack_layout_b.grid, cosma_grid_b);

    if (std::abs(beta) > 0) {
        comm_vol += grid2grid::communication_volume(scalapack_layout_c.grid, cosma_grid_c);
    }

    comm_vol += grid2grid::communication_volume(cosma_grid_c, scalapack_layout_c.grid);

    // compute the optimal rank reordering that minimizes the communication volume
    bool reordered = false;
    std::vector<int> rank_permutation = grid2grid::optimal_reordering(comm_vol, P, reordered);
    PL();

#ifdef DEBUG
    if (rank == 0) {
        std::cout << "Optimal rank relabeling:" << std::endl;
        for (int i = 0; i < P; ++i) {
            std::cout << i << "->" << rank_permutation[i] << std::endl;
        }
    }
#endif

    CosmaMatrix<T> A(std::move(mapper_a), rank_permutation[rank]);
    CosmaMatrix<T> B(std::move(mapper_b), rank_permutation[rank]);
    CosmaMatrix<T> C(std::move(mapper_c), rank_permutation[rank]);

    // avoid resizing of buffer by reserving immediately the total required memory
    get_context_instance<T>()->get_memory_pool().reserve(A.total_required_memory()
                                                       + B.total_required_memory()
                                                       + C.total_required_memory());

    // get abstract layout descriptions for COSMA layout
    auto cosma_layout_a = A.get_grid_layout();
    auto cosma_layout_b = B.get_grid_layout();
    auto cosma_layout_c = C.get_grid_layout();

    cosma_layout_a.reorder_ranks(rank_permutation);
    cosma_layout_b.reorder_ranks(rank_permutation);
    cosma_layout_c.reorder_ranks(rank_permutation);

    // if (rank == 0) {
    //     std::cout << "COSMA grid for A after reordering: " << cosma_layout_a.grid << std::endl;
    // }

#ifdef DEBUG
    std::cout << "Transforming the input matrices A and B from Scalapack -> COSMA" << std::endl;
#endif
    // transform A and B from scalapack to cosma layout
    grid2grid::transformer<T> transf(comm);
    transf.schedule(scalapack_layout_a, cosma_layout_a);
    transf.schedule(scalapack_layout_b, cosma_layout_b);

    // transform C from scalapack to cosma only if beta > 0
    if (std::abs(beta) > 0) {
        transf.schedule(scalapack_layout_c, cosma_layout_c);
    }

    // transform all scheduled transformations together
    transf.transform();

#ifdef DEBUG
    std::cout << "COSMA multiply" << std::endl;
#endif
    // create reordered communicator, which has same ranks
    // but relabelled as given by the rank_permutation
    // (to avoid the communication during layout transformation)
    PE(transform_reordering_comm);
    MPI_Comm reordered_comm = comm;
    if (reordered) {
        MPI_Comm_split(comm, 0, rank_permutation[rank], &reordered_comm);
    }
    PL();
    // perform cosma multiplication
    multiply<T>(A, B, C, strategy, reordered_comm, alpha, beta);
    PE(transform_reordering_comm);
    if (reordered) {
        MPI_Comm_free(&reordered_comm);
    }
    PL();

    // construct cosma layout again, to avoid outdated
    // pointers when the memory pool has been used
    // in case it resized during multiply
    cosma_layout_c = C.get_grid_layout();
    cosma_layout_c.reorder_ranks(rank_permutation);

#ifdef DEBUG
    std::cout << "Transforming the result C back from COSMA to ScaLAPACK" << std::endl;
#endif
    // grid2grid::transform the result from cosma back to scalapack
    // grid2grid::transform<T>(cosma_layout_c, scalapack_layout_c, comm);
    transf.schedule(cosma_layout_c, scalapack_layout_c);
    transf.transform();

#ifdef DEBUG
    if (rank == 0) {
        auto reordered_vol = grid2grid::communication_volume(scalapack_layout_a.grid, cosma_layout_a.grid);
        reordered_vol += grid2grid::communication_volume(scalapack_layout_b.grid, cosma_layout_b.grid);
        if (std::abs(beta) > 0) {
            reordered_vol += grid2grid::communication_volume(scalapack_layout_c.grid, cosma_layout_c.grid);
        }
        reordered_vol += grid2grid::communication_volume(cosma_layout_c.grid, scalapack_layout_c.grid);

        // std::cout << "Detailed comm volume: " << comm_vol << std::endl;
        // std::cout << "Detailed comm volume reordered: " << reordered_vol << std::endl;

        auto comm_vol_total = comm_vol.total_volume();
        auto reordered_vol_total = reordered_vol.total_volume();
        std::cout << "Initial comm volume = " << comm_vol_total << std::endl;
        std::cout << "Reduced comm volume = " << reordered_vol_total << std::endl;
        auto diff = (long long) comm_vol_total - (long long) reordered_vol_total;
        std::cout << "Comm volume reduction [%] = " << 100.0 * diff / comm_vol_total << std::endl;

    }
#endif
}

// explicit instantiation for pxgemm
template void pxgemm<double>(const char trans_a,
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

template void pxgemm<float>(const char trans_a,
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

template void pxgemm<zdouble_t>(const char trans_a,
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

template void pxgemm<zfloat_t>(const char trans_a,
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
} // namespace cosma

