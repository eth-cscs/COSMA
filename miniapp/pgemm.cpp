#include <cosma/pdgemm_wrapper.hpp>

using namespace cosma;

template <typename T>
void fillInt(T &in) {
    std::generate(in.begin(), in.end(), []() { return (int)(10 * drand48()); });
}

int get_numroc(int n, int nb, int iproc, int isrcproc, int nprocs) {
    // -- ScaLAPACK tools routine (version 1.7) --
    //    University of Tennessee, Knoxville, Oak Ridge National Laboratory,
    //    and University of California, Berkeley.
    //    May 1, 1997
    //
    // Purpose
    // =======
    //
    // NUMROC computes the NUMber of Rows Or Columns of a distributed
    // matrix owned by the process indicated by IPROC.
    //
    // Arguments
    // =========
    //
    // N         (global input) INTEGER
    //           The number of rows/columns in distributed matrix.
    //
    // NB        (global input) INTEGER
    //           Block size, size of the blocks the distributed matrix is
    //           split into.
    //
    // IPROC     (local input) INTEGER
    //           The coordinate of the process whose local array row or
    //           column is to be determined.
    //
    // ISRCPROC  (global input) INTEGER
    //           The coordinate of the process that possesses the first
    //           row or column of the distributed matrix.
    //
    // NPROCS    (global input) INTEGER
    //           The total number processes over which the matrix is
    //           distributed.

    // Figure PROC's distance from source process
    int mydist = (nprocs + iproc - isrcproc) % nprocs;

    // Figure the total number of whole NB blocks N is split up into
    int nblocks = n / nb;

    // Figure the minimum number of rows/cols a process can have
    int return_value = (nblocks / nprocs) * nb;

    // See if there are any extra blocks
    int extrablks = nblocks % nprocs;

    // If I have an extra block
    if (mydist < extrablks)
        return_value += nb;
    // If I have last block, it may be a partial block
    else if (mydist == extrablks)
        return_value += n % nb;

    return return_value;
}


long run(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Initialize Cblas context */
    int ctxt, myid, myrow, mycol, numproc;
    // assume we have 2x2 processor grid
    int procrows = 2, proccols = 2;
    std::cout << "Cblacs_pinfo" << std::endl;
    Cblacs_pinfo(&myid, &numproc);
    std::cout << "Cblacs_get" << std::endl;
    Cblacs_get(0, 0, &ctxt);
    char order = 'R';
    std::cout << "Cblacs_gridinit" << std::endl;
    Cblacs_gridinit(&ctxt, &order, procrows, proccols);
    std::cout << "Cblacs_pcoord" << std::endl;
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    std::cout << "Rank = " << rank << ", myid = " << myid << std::endl;
    std::cout << "My pcoord = " << myrow << ", " << mycol << std::endl;

    // describe a problem size
    int m = 10;
    int n = 10;
    int k = 10;

    int bm = 2;
    int bn = 2;
    int bk = 2;

    char trans_a = 'N';
    char trans_b = 'N';
    char trans_c = 'N';

    int alpha = 1.0;
    int beta = 0.0;

    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = 1;
    int ic = 1;
    int jc = 1;

    int rsrc = 0;
    int csrc = 0;

    int iZERO = 0;

    std::cout << "numroc A, rows" << std::endl;
    int nrows_a = get_numroc(m+ia-1, bm, myrow, rsrc, procrows);
    std::cout << "numroc B, rows" << std::endl;
    int nrows_b = get_numroc(k+ib-1, bk, myrow, rsrc, procrows);
    std::cout << "numroc C, rows" << std::endl;
    int nrows_c = get_numroc(m+ic-1, bm, myrow, rsrc, procrows);

    std::cout << "numroc A, cols" << std::endl;
    int ncols_a = get_numroc(k+ja-1, bk, mycol, csrc, proccols);
    std::cout << "numroc B, cols" << std::endl;
    int ncols_b = get_numroc(n+jb-1, bn, mycol, csrc, proccols);
    std::cout << "numroc C, cols" << std::endl;
    int ncols_c = get_numroc(n+jc-1, bn, mycol, csrc, proccols);

    std::cout << "Initializing ScaLAPACK buffers for A, B and C" << std::endl;
    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c(nrows_c * ncols_c);

    // initialize matrices A, B and C
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    std::cout << "descinit A" << std::endl;
    descinit(&desc_a[0], &m, &k, &bm, &bk, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    std::cout << "descinit B" << std::endl;
    descinit(&desc_b[0], &k, &n, &bk, &bn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    std::cout << "descinit C" << std::endl;
    descinit(&desc_c[0], &m, &n, &bm, &bn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    std::cout << "Filling up ScaLAPACK matrices with random data." << std::endl;
    srand48(rank);
    fillInt(a);
    fillInt(b);
    fillInt(c);

    std::cout << "Invoking pdgemm_wrapper" << std::endl;
    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    pgemm<double>(trans_a, trans_b, m, n, k,
           alpha, a.data(), ia, ja, &desc_a[0],
           b.data(), ib, jb, &desc_b[0], beta,
           c.data(), ic, jc, &desc_c[0]);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Finished pdgemm_wrapper" << std::endl;

    std::cout << "Cblacs_gridexit" << std::endl;
    Cblacs_gridexit(ctxt);
    std::cout << "Cblacs_exit" << std::endl;
    Cblacs_exit(EXIT_SUCCESS);

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    std::cout << "Initialized MPI." << std::endl;

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_iter = 1;
    std::vector<long> times;
    for (int i = 0; i < n_iter; ++i) {
        long t_run = 0;
        t_run = run();
        times.push_back(t_run);
    }
    std::sort(times.begin(), times.end());

    if (rank == 0) {
        std::cout << "PDGEMM_WRAPPER TIMES [ms] = ";
        for (auto &time : times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
