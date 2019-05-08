#include <mpi.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <array>
#include <numeric>

#include <layout_transformer/transform.hpp>
#include <layout_transformer/scalapack_layout.hpp>
#include <math_utils.hpp>

using namespace layout_transformer;

extern "C" {
    /* Cblacs declarations */
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, const char*, int, int);
    void Cblacs_pcoord(int, int, int*, int*);
    void Cblacs_gridexit(int);
    void Cblacs_barrier(int, const char*);

    int numroc_(int*, int*, int*, int*, int*);

    void pdgemr2d_(int *m, int *n,
            double *a, int *ia, int *ja, int *desca,
            double *b, int *ib, int *jb, int *descb,
            int* ictxt);

    void descinit_(int* desc, int* m, int* n, int* bm, int* bn, 
            int* rsrc, int* csrc, int* ctxt, int* lda, int* info);
}

// *****************************
// OUR LAYOUT TRANSFORMER
// *****************************
long int run_our_layout(int m, int n, int bm1, int bn1, int bm2, int bn2, int pm, int pn, int nrep, int rank) {
    auto ordering = scalapack::ordering::row_major;

    auto values = [](int i, int j) {
        return cosma::math_utils::cantor_pairing(i, j);
    };

    scalapack::data_layout layout1({m, n}, {bm1, bn1}, {pm, pn}, ordering);
    std::vector<double> buffer1 = initialize_locally(rank, layout1, values);
    grid_layout scalapack_layout_1 = get_scalapack_grid(layout1, buffer1.data(), rank);

    scalapack::data_layout layout2({m, n}, {bm2, bn2}, {pm, pn}, ordering);
    std::vector<double> buffer2 = initialize_locally(rank, layout2, values);
    grid_layout scalapack_layout_2 = get_scalapack_grid(layout2, buffer2.data(), rank);

    long int min_time = std::numeric_limits<long int>::max();

    for (int i = 0; i < nrep; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();

        transform(scalapack_layout_1, scalapack_layout_2, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();

        auto our_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(our_time, min_time);
    }

    return min_time;
}

// *****************************
// SCALAPACK LAYOUT TRANSFORMER
// *****************************
long int run_scalapack_layout(int m, int n, int bm1, int bn1, int bm2, int bn2, int pm, int pn, int nrep, int rank) {
    // Begin Cblas context
    // We assume that we have 4 processes and place them in a 2-by-2 grid
    int iZERO = 0;
    int ctxt, myid, myrow, mycol, numproc;
    int procrows = 2, proccols = 2;

    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    Cblacs_gridinit(&ctxt, "Row-major", procrows, proccols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);

    // Number of rows and cols owned by the current process
    int nrows1 = numroc_(&m, &bm1, &myrow, &iZERO, &procrows);
    int ncols1 = numroc_(&n, &bn1, &mycol, &iZERO, &proccols);

    int nrows2 = numroc_(&m, &bm2, &myrow, &iZERO, &procrows);
    int ncols2 = numroc_(&n, &bn2, &mycol, &iZERO, &proccols);

    std::vector<double> buffer1(nrows1 * ncols1);
    std::vector<double> buffer2(nrows2 * ncols2);

    int ia = 1;
    int ja = 1;
    int ib = 1;
    int jb = 1;

    // std::vector<int> desca = {1, ctxt, m, n, bm1, bn1, 0, 0, m};
    // std::vector<int> descb = {1, ctxt, m, n, bm2, bn2, 0, 0, m};

    std::array<int, 9> desc1;
    std::array<int, 9> desc2;
    int info;
    descinit_(&desc1[0], &m, &n, &bm1, &bn1, &iZERO, &iZERO, &ctxt, &nrows1, &info);
    descinit_(&desc2[0], &m, &n, &bm2, &bn2, &iZERO, &iZERO, &ctxt, &nrows2, &info);

    long int min_time = std::numeric_limits<long int>::max();
    for (int i = 0; i < nrep; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();
        pdgemr2d_(&m, &n, buffer1.data(), &ia, &ib, &desc1[0],
                buffer2.data(), &ib, &jb, &desc2[0],
                &ctxt);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto scalapack_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        min_time = std::min(min_time, scalapack_time);
    }

    // Release resources
    Cblacs_gridexit(ctxt);

    return min_time;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim = 10000;
    int bm1 = 124;
    int bn1 = 124;

    int pm = 2;
    int pn = 2;

    int bm2 = 192;
    int bn2 = 192;

    int nrep = 3;

    for (int i = 1; i <= 5; ++i) { 
        int m = dim * i;
        int n = dim * i;

        auto our_time = run_our_layout(m, n, bm1, bn1, bm2, bn2, pm, pn, nrep, rank);
        auto scalapack_time = run_scalapack_layout(m, n, bm1, bn1, bm2, bn2, pm, pn, nrep, rank);

        if (rank == 0) {
            std::cout << "Dimension = " << m << std::endl;
            std::cout << "Our time [ms] = " << our_time << std::endl;
            std::cout << "ScaLAPACK time [ms] = " << scalapack_time << std::endl;
            std::cout << "Ration scalapack/our = " << 1.0 * scalapack_time/our_time << std::endl;
            std::cout << "============================" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

