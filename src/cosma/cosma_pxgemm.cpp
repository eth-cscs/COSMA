#include <cassert>
#include <mpi.h>

#include <cosma/blacs.hpp>
#include <cosma/multiply.hpp>
#include <cosma/cosma_pxgemm.hpp>
#include <cosma/profiler.hpp>
#include <cosma/pxgemm_params.hpp>
#include <cosma/environment_variables.hpp>

#include <costa/grid2grid/ranks_reordering.hpp>
#include <costa/grid2grid/transformer.hpp>

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
    // **********************************
    //           CORNER CASES
    // **********************************
    // edge cases, which are allowed by the standard
    if (m == 0 || n == 0) return;
    // afterwards we are sure m != 0 and n != 0
    if (k == 0 || alpha == T{0}) {
        // scale matrix C by beta
        // starting from (ic-1, jc-1)
        scale_matrix(descc, c, ic, jc, m, n, beta);
        return;
    }
    // afterwards we are sure k != 0 and alpha != 0
    // case beta == 0 is already handled by the code below

    // **********************************
    //           MAIN CODE
    // **********************************
    // clear the profiler
    PC();
    // start profiling
    PE(init);
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
    char grid_order = 
        ordering == costa::scalapack::ordering::column_major ? 'C' : 'R';

#ifdef DEBUG
    if (rank == 0) {
        pxgemm_params<T> params(
                             // global dimensions
                             mat_dim_a.rows, mat_dim_a.cols,
                             mat_dim_b.rows, mat_dim_b.cols,
                             mat_dim_c.rows, mat_dim_c.cols,
                             // block dimensions
                             b_dim_a.rows, b_dim_a.cols,
                             b_dim_b.rows, b_dim_b.cols,
                             b_dim_c.rows, b_dim_c.cols,
                             // submatrix start
                             ia, ja,
                             ib, jb,
                             ic, jc,
                             // problem size
                             m, n, k,
                             // transpose flags
                             trans_a, trans_b,
                             // alpha, beta
                             alpha, beta,
                             // leading dimensinons
                             lld_a, lld_b, lld_c,
                             // processor grid
                             procrows, proccols,
                             // processor grid ordering
                             grid_order,
                             // ranks containing first rows
                             rank_src_a.row_src, rank_src_a.col_src,
                             rank_src_b.row_src, rank_src_b.col_src,
                             rank_src_c.row_src, rank_src_c.col_src
                         );
        std::cout << params << std::endl;
    }
    MPI_Barrier(comm);
#endif

    std::vector<int> divisors;
    std::string step_type = "";
    std::string dimensions = "";
    PL();

    PE(strategy);
    /*
      If the matrix is very large, then its reshuffling is expensive.
      For this reason, try to adapt the strategy to the scalapack layout
      to minimize the need for reshuffling, even if it makes a 
      suoptimal communication scheme in COSMA.
      This method will add "prefix" to the strategy, i.e. some initial steps
      that COSMA should start with and then continue with finding 
      the communication-optimal strategy.
     */
    bool strategy_adapted = false;
    if (P > 1 && get_context_instance<T>()->adapt_to_scalapack_strategy) {
        adapt_strategy_to_block_cyclic_grid(divisors, dimensions, step_type,
                                            m, n, k, P,
                                            mat_dim_a, mat_dim_b, mat_dim_c,
                                            b_dim_a, b_dim_b, b_dim_c,
                                            ia, ja, ib, jb, ic, jc,
                                            trans_a, trans_b,
                                            procrows, proccols,
                                            grid_order
                                            );
        if (step_type != "") {
            strategy_adapted = true;
        }
    }

    // get CPU memory limit
    auto cpu_memory_limit = get_context_instance<T>()->get_cpu_memory_limit();
    Strategy strategy(m, n, k, P,
                      divisors, dimensions, step_type,
                      cpu_memory_limit);
    // enable overlapping communication and computation if turned on
    if (get_context_instance<T>()->overlap_comm_and_comp) {
        strategy.enable_overlapping_comm_and_comp();
    }
    PL();

    PE(init);

#ifdef DEBUG
    if (rank == 0) {
        std::cout << strategy << std::endl;
        std::cout << "============================================" << std::endl;
    }
    MPI_Barrier(comm);
#endif

    PL();
    // create COSMA mappers
    Mapper mapper_a('A', strategy, rank);
    Mapper mapper_b('B', strategy, rank);
    Mapper mapper_c('C', strategy, rank);

    auto cosma_grid_a = mapper_a.get_layout_grid();
    auto cosma_grid_b = mapper_b.get_layout_grid();
    auto cosma_grid_c = mapper_c.get_layout_grid();

    PE(transform_init);
    // get abstract layout descriptions for ScaLAPACK layout
    auto scalapack_layout_a = costa::get_scalapack_layout<T>(
        lld_a,
        {mat_dim_a.rows, mat_dim_a.cols},
        {ia, ja},
        {a_subm, a_subn},
        {b_dim_a.rows, b_dim_a.cols},
        {procrows, proccols},
        ordering,
        {rank_src_a.row_src, rank_src_a.col_src},
        a,
        'C',
        rank);

    auto scalapack_layout_b = costa::get_scalapack_layout<T>(
        lld_b,
        {mat_dim_b.rows, mat_dim_b.cols},
        {ib, jb},
        {b_subm, b_subn},
        {b_dim_b.rows, b_dim_b.cols},
        {procrows, proccols},
        ordering,
        {rank_src_b.row_src, rank_src_b.col_src},
        b,
        'C',
        rank);

    auto scalapack_layout_c = costa::get_scalapack_layout<T>(
        lld_c,
        {mat_dim_c.rows, mat_dim_c.cols},
        {ic, jc},
        {c_subm, c_subn},
        {b_dim_c.rows, b_dim_c.cols},
        {procrows, proccols},
        ordering,
        {rank_src_c.row_src, rank_src_c.col_src},
        c,
        'C',
        rank);
    PL();

    // by default, no process-relabeling is assumed.
    bool reordered = false;
    std::vector<int> rank_permutation;
    MPI_Comm reordered_comm = comm;

    if (!strategy_adapted) {
        PE(transform_reordering_matching);
        // total communication volume for transformation of layouts
        // costa::comm_volume comm_vol;
        auto comm_vol = costa::communication_volume(scalapack_layout_a.grid, cosma_grid_a, trans_a);
        comm_vol += costa::communication_volume(scalapack_layout_b.grid, cosma_grid_b, trans_b);
        comm_vol += costa::communication_volume(cosma_grid_c, scalapack_layout_c.grid, 'N');

        // compute the optimal rank reordering that minimizes the communication volume
        rank_permutation = costa::optimal_reordering(comm_vol, P, reordered);
        PL();

        // create reordered communicator, which has same ranks
        // but relabelled as given by the rank_permutation
        // (to avoid the communication during layout transformation)
        PE(transform_reordering_comm);
        if (reordered) {
            MPI_Comm_split(comm, 0, rank_permutation[rank], &reordered_comm);
        }
        PL();
    } else {
        rank_permutation.reserve(P);
        // if the strategy is adapted, then no process-relabeling occurs.
        for (int i = 0; i < P; ++i) {
            rank_permutation.push_back(i);
        }
    }

#ifdef DEBUG
    if (rank == 0) {
        std::cout << "Optimal rank relabeling:" << std::endl;
        for (int i = 0; i < P; ++i) {
            std::cout << i << "->" << rank_permutation[i] << std::endl;
        }
    }
#endif

    // first, we don't want to alloate the space, just to precompute
    // the required memory size, so we active dry_run, which precomputes
    // everything but doesn't allocate anything yet
    bool dont_allocate = true;
    CosmaMatrix<T> A(std::move(mapper_a), rank_permutation[rank], dont_allocate);
    CosmaMatrix<T> B(std::move(mapper_b), rank_permutation[rank], dont_allocate);
    CosmaMatrix<T> C(std::move(mapper_c), rank_permutation[rank], dont_allocate);

    // avoid resizing the buffer by reserving immediately the total required memory
    // collect sizes of all buffers that are going to be allocated for each matrix
    auto A_buffers = A.required_memory();
    auto B_buffers = B.required_memory();
    auto C_buffers = C.required_memory();

    std::vector<std::size_t> buffer_sizes;
    int n_buffers = A_buffers.size() + B_buffers.size() + C_buffers.size();
    if (n_buffers > 0) {
        buffer_sizes.reserve(n_buffers);
        std::copy(A_buffers.begin(), A_buffers.end(), std::back_inserter(buffer_sizes));
        std::copy(B_buffers.begin(), B_buffers.end(), std::back_inserter(buffer_sizes));
        std::copy(C_buffers.begin(), C_buffers.end(), std::back_inserter(buffer_sizes));

        // allocate all buffers in the memory pool
        get_context_instance<T>()->get_memory_pool().reserve(buffer_sizes);
    }

    // turn off dryrun mode, allocate memory for all matrices
    A.allocate();
    B.allocate();
    C.allocate();

    // get abstract layout descriptions for COSMA layout
    auto cosma_layout_a = A.get_grid_layout();
    auto cosma_layout_b = B.get_grid_layout();
    auto cosma_layout_c = C.get_grid_layout();

    cosma_layout_a.reorder_ranks(rank_permutation);
    cosma_layout_b.reorder_ranks(rank_permutation);
    cosma_layout_c.reorder_ranks(rank_permutation);

#ifdef DEBUG
    std::cout << "Transforming the input matrices A and B from Scalapack -> COSMA" << std::endl;
#endif
    // transform A and B from scalapack to cosma layout
    costa::transformer<T> transf(comm);
    transf.schedule(scalapack_layout_a, cosma_layout_a, trans_a, T{1}, T{0});
    // transf.transform();
    transf.schedule(scalapack_layout_b, cosma_layout_b, trans_b, T{1}, T{0});

    transf.transform();

#ifdef DEBUG
    std::cout << "COSMA multiply" << std::endl;
#endif

    // perform cosma multiplication
    multiply<T>(A, B, C, strategy, reordered_comm, T{1}, T{0});
    // construct cosma layout again, to avoid outdated
    // pointers when the memory pool has been used
    // in case it resized during multiply
    cosma_layout_c = C.get_grid_layout();
    cosma_layout_c.reorder_ranks(rank_permutation);

#ifdef DEBUG
    std::cout << "Transforming the result C back from COSMA to ScaLAPACK" << std::endl;
#endif
    // costa::transform the result from cosma back to scalapack
    // costa::transform<T>(cosma_layout_c, scalapack_layout_c, comm);
    transf.schedule(cosma_layout_c, scalapack_layout_c, 'N', alpha, beta);

    transf.transform();

#ifdef DEBUG
    if (rank == 0) {
        auto reordered_vol = costa::communication_volume(scalapack_layout_a.grid, cosma_layout_a.grid);
        reordered_vol += costa::communication_volume(scalapack_layout_b.grid, cosma_layout_b.grid);
        if (std::abs(beta) > 0) {
            reordered_vol += costa::communication_volume(scalapack_layout_c.grid, cosma_layout_c.grid);
        }
        reordered_vol += costa::communication_volume(cosma_layout_c.grid, scalapack_layout_c.grid);

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

    PE(transform_reordering_comm);
    if (reordered) {
        MPI_Comm_free(&reordered_comm);
    }
    PL();
}

// scales the submatrix of C by beta
// The submatrix is defined by (ic-1, jc-1) and (ic-1+m, jc-1+n)
template <typename T>
void scale_matrix(const int* descc, T* c,
                  const int ic, const int jc,
                  const int m, const int n,
                  const T beta) {
    if (beta == T{1}) return;
    // clear the profiler
    PC();

    // start profiling
    PE(init);

    // blas context
    int ctxt = scalapack::get_grid_context(descc);

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
    scalapack::block_size b_dim_c(descc);

    // global matrix sizes
    scalapack::global_matrix_size mat_dim_c(descc);

    // sumatrix size to multiply
    int c_subm = m;
    int c_subn = n;

    // rank sources (rank coordinates that own first row and column of a matrix)
    scalapack::rank_src rank_src_c(descc);

    // leading dimensions
    int lld_c = scalapack::leading_dimension(descc);

    // check whether rank grid is row-major or col-major
    auto ordering = scalapack::rank_ordering(ctxt, P);
    char grid_order = 
        ordering == costa::scalapack::ordering::column_major ? 'C' : 'R';

    // create costa object describing the given scalapack layout
    auto layout = costa::get_scalapack_layout<T>(
        lld_c,
        {mat_dim_c.rows, mat_dim_c.cols},
        {ic, jc},
        {c_subm, c_subn},
        {b_dim_c.rows, b_dim_c.cols},
        {procrows, proccols},
        ordering,
        {rank_src_c.row_src, rank_src_c.col_src},
        c,
        'C',
        rank);
    PL();

    PE(multiply_computation);
    // scale the elements in the submatrix given by the layout
    layout.scale_by(beta);
    PL();
}

// returns A, B or C, depending on which flag was set to true.
// used to minimize the number of if-else statements in adapt_strategy
template <typename T>
T& one_of(T &A,
         T &B,
         T &C,
         bool first,
         bool second,
         bool third) {
    if (first) return A;
    if (second) return B;
    return C;
}

// returns the largest of (first, second, third) and sets the corresponding
// boolean flag of the largest element to true.
// used to minimize the number of if-else statements in adapt_strategy
template <typename T>
T which_is_largest(T&& first, T&& second, T&& third,
                      bool& first_largest, bool& second_largest, bool& third_largest) {
    T largest = std::max(std::max(first, second), third);
    first_largest = false;
    second_largest = false;
    third_largest = false;
    if (largest == first) {
        first_largest = true;
        return std::forward<T>(first);
    }
    if (largest == second) {
        second_largest = true;
        return std::forward<T>(second);
    }
    if (largest == third) {
        third_largest = true;
        return std::forward<T>(third);
    }
    return T{};
}

char get_matrix_dimension(bool matrix_A, bool matrix_B, bool matrix_C,
                                 char trans_a, char trans_b,
                                 int index) {
    std::string dimensions = "";
    if (matrix_A) {
        // if transposed
        dimensions = trans_a != 'N' ? "km" : "mk";
    } else if (matrix_B) {
        dimensions = trans_b != 'N' ? "kn" : "nk";
    } else {
        dimensions = "mn";
    }

    return dimensions[index];
}

void adapt_strategy_to_block_cyclic_grid(// these will contain the suggested strategy prefix
                                         std::vector<int>& divisors, 
                                         std::string& dimensions,
                                         std::string& step_type,
                                         // multiplication problem size
                                         int m, int n, int k, int P,
                                         // global matrix dimensions
                                         scalapack::global_matrix_size& mat_dim_a,
                                         scalapack::global_matrix_size& mat_dim_b,
                                         scalapack::global_matrix_size& mat_dim_c,
                                         // block sizes
                                         scalapack::block_size& b_dim_a,
                                         scalapack::block_size& b_dim_b,
                                         scalapack::block_size& b_dim_c,
                                         // (i, j) denoting the submatrix coordinates
                                         int ia, int ja,
                                         int ib, int jb,
                                         int ic, int jc,
                                         // transpose flags
                                         char trans_a, char trans_b,
                                         // processor grid
                                         int procrows, int proccols,
                                         char order
                                         ) {
    // If the matrix is very large, then its reshuffling is expensive.
    // For this reason, try to adapt the strategy to the scalapack layout
    // to minimize the need for reshuffling, even if it makes a 
    // suoptimal communication scheme in COSMA.
    // Here, we only do this optimization if the scalapack grid
    // fills up the matrix completely (everything is perfectly divisible).
    // Since there are 3 matrices, we only focus on the largest one.
    bool first = false;
    bool second = false;
    bool third = false;

    // sumatrix size to multiply
    int a_subm = trans_a == 'N' ? m : k;
    int a_subn = trans_a == 'N' ? k : m;

    int b_subm = trans_b == 'N' ? k : n;
    int b_subn = trans_b == 'N' ? n : k;

    int c_subm = m;
    int c_subn = n;

    long long largest_matrix_local_size = 
        which_is_largest(1LL * m * k, 
                         1LL * k * n, 
                         1LL * m * n, 
                         first,
                         second,
                         third) / P;

    // We only apply this optimization if the matrix is large enough,
    // because adapting the strategy to the given initial grid
    // might result in a communication-suboptimal strategy.
    // However, when the reshuffling cost is too high, then it might be beneficial
    // to make COSMA use a communication-suboptimal strategy
    // to reduce the overall time.
    if (largest_matrix_local_size > 1e7) {
        auto b_dim = one_of(b_dim_a, b_dim_b, b_dim_c, first, second, third);
        auto mat_dim = one_of(mat_dim_a, mat_dim_b, mat_dim_c, first, second, third);
        auto subm = one_of(a_subm, b_subm, c_subm, first, second, third);
        auto subn = one_of(a_subn, b_subn, c_subn, first, second, third);
        auto i = one_of(ia, ib, ic, first, second, third);
        auto j = one_of(ja, jb, jc, first, second, third);

        // The whole matrix should take part in the multiplication,
        // the blocks sizes should perfectly divide the matrix
        // and processor grid must perfectly cover the matrix blocks grid.
        if ((i == 1 && j == 1)  // no submatrix
            && (subm == mat_dim.rows && subn == mat_dim.cols) // no submatrix
            && (mat_dim.rows % b_dim.rows == 0) // blocks perfectly divide the matrix
            && (mat_dim.cols % b_dim.cols == 0)  // blocks perfectly divide the matrix
            && (mat_dim.rows / b_dim.rows % procrows == 0)
            && (mat_dim.cols / b_dim.cols % proccols == 0)) // processor grid divides the matrix blocks grid
        {
            int divisor_rows = mat_dim.rows / b_dim.rows / procrows;
            int divisor_cols = mat_dim.cols / b_dim.cols / proccols;

            // adding sequential steps
            if (divisor_rows > 1) {
                step_type += "s";
                divisors.push_back(divisor_rows);
                dimensions += get_matrix_dimension(first, second, third,
                                                   trans_a, trans_b,
                                                   0); // 0 means rows
            }

            if (divisor_cols > 1) {
                step_type += "s";
                divisors.push_back(divisor_cols);
                dimensions += get_matrix_dimension(first, second, third,
                                                   trans_a, trans_b,
                                                   1); // 1 means columns
            }

            // adding parallel steps
            if (order == 'R') {
                // first add rows split and then cols split if applicable
                if (procrows > 1) {
                    step_type += "p";
                    divisors.push_back(procrows);
                    dimensions += get_matrix_dimension(first, second, third,
                                                       trans_a, trans_b,
                                                       0); // 1 means columns
                }
                if (proccols > 1) {
                    step_type += "p";
                    divisors.push_back(proccols);
                    dimensions += get_matrix_dimension(first, second, third,
                                                       trans_a, trans_b,
                                                       1); // 1 means columns
                }
            } else {
                // first add cols split and then rows split if applicable
                if (proccols > 1) {
                    step_type += "p";
                    divisors.push_back(proccols);
                    dimensions += get_matrix_dimension(first, second, third,
                                                       trans_a, trans_b,
                                                       1); // 1 means columns
                }
                if (procrows > 1) {
                    step_type += "p";
                    divisors.push_back(procrows);
                    dimensions += get_matrix_dimension(first, second, third,
                                                       trans_a, trans_b,
                                                       0); // 1 means columns
                }
            }
        }
    }
}

bool is_problem_too_small(int m, int n, int k) {
    static const int cosma_dim_threshold = cosma::get_cosma_dim_threshold();
    return std::min(m, std::min(n, k)) < cosma_dim_threshold;
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

