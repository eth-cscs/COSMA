#include <cosma/pdgemm_wrapper.hpp>

using namespace cosma;

template <typename T>
void fillInt(T &in) {
    std::generate(in.begin(), in.end(), []() { return (int)(10 * drand48()); });
}

// Reads an environment variable `n_iter`
//
int get_n_iter() {
    int intValue = std::atoi(std::getenv("n_iter"));
    if (intValue < 1 || intValue > 100) {
        std::cout << "Number of iteration must be in the interval [1, 100]"
                  << std::endl;
        std::cout << "Setting it to 1 iteration instead" << std::endl;
        return 1;
    }

    return intValue;
}

long run(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Initialize Cblas context */
    int ctxt, myid, myrow, mycol, numproc;
    // assume we have 2x2 processor grid
    int procrows = 2, proccols = 2;
    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    char order = 'R';
    Cblacs_gridinit(&ctxt, &order, procrows, proccols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    // describe a problem size
    int m = 1000;
    int n = 1000;
    int k = 1000;

    int bm = 128;
    int bn = 128;
    int bk = 128;

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

    int nrows_a = numroc(m, bm, myrow, rsrc, procrows);
    int nrows_b = numroc(k, bk, myrow, rsrc, procrows);
    int nrows_c = numroc(m, bm, myrow, rsrc, procrows);

    int ncols_a = numroc(k, bk, mycol, csrc, proccols);
    int ncols_b = numroc(n, bn, mycol, csrc, proccols);
    int ncols_c = numroc(n, bn, mycol, csrc, proccols);

    std::vector<double> a(nrows_a * ncols_a);
    std::vector<double> b(nrows_b * ncols_b);
    std::vector<double> c(nrows_c * ncols_c);

    // initialize matrices A, B and C
    std::array<int, 9> desc_a;
    std::array<int, 9> desc_b;
    std::array<int, 9> desc_c;
    int info;
    Cblacs_descinit(&desc_a[0], &m, &k, &bm, &bk, &rsrc, &csrc, &ctxt, &nrows_a, &info);
    Cblacs_descinit(&desc_b[0], &k, &n, &bk, &bn, &rsrc, &csrc, &ctxt, &nrows_b, &info);
    Cblacs_descinit(&desc_c[0], &m, &n, &bm, &bn, &rsrc, &csrc, &ctxt, &nrows_c, &info);

    // fill the matrices with random data
    srand48(rank);

    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    pgemm<double>(trans_a, trans_b, m, n, k,
           alpha, a.data(), ia, ja, &desc_a[0],
           b.data(), ib, jb, &desc_b[0], beta,
           c.data(), ic, jc, &desc_c[0]);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();

    Cblacs_gridexit(ctxt);
    Cblacs_exit(EXIT_SUCCESS);

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_iter = get_n_iter();
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
