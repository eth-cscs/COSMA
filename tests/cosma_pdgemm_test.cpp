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
#include "cosma_pdgemm_run.hpp"
#include <cosma/blacs.hpp>
#include <cosma/pgemm.hpp>

// from options
#include <options.hpp>

using namespace cosma;

int main(int argc, char **argv) {
    // **************************************
    //   setup MPI and command-line parser
    // **************************************
    options::initialize(argc, argv);

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // **************************************
    //   readout the command line arguments
    // **************************************
    // matrix dimensions
    // dim(A) = mxk, dim(B) = kxn, dim(C) = mxn
    auto m = options::next_int("-m", "--m_dim", "number of rows of A and C.", 1000);
    auto n = options::next_int("-n", "--n_dim", "number of columns of B and C.", 1000);
    auto k = options::next_int("-k", "--k_dim", "number of columns of A and rows of B.", 1000);

    // block sizes
    auto bm = options::next_int("-bm", "--m_block", "block size for the number of rows of A and C.", 128);
    auto bn = options::next_int("-bn", "--n_block", "block size for the number of columns of B and C.", 128);
    auto bk = options::next_int("-bk", "--k_block", "block size for the number of columns of A and rows of B.", 128);

    // transpose flags
    bool trans_a = options::flag_exists("-ta", "--trans_a");
    bool trans_b = options::flag_exists("-tb", "--trans_b");
    char ta = trans_a ? 'T' : 'N';
    char tb = trans_b ? 'T' : 'N';

    // processor grid decomposition
    auto p = options::next_int("-p", "--p_row", "number of rows in a processor grid.", 1);
    auto q = options::next_int("-q", "--q_row", "number of columns in a processor grid.", P);

    if (p * q != P) {
        std::runtime_error("Number of processors in a grid has to match the number of available ranks.");
    }

    double alpha = 1.0;
    double beta = 0.0;

    // **************************************
    //    output the problem description
    // **************************************
    if (rank == 0) {
        std::cout << "Running PDGEMM on the following problem size:" << std::endl;
        std::cout << "Matrix sizes: (m, n, k) = (" << m << ", " << n << ", " << k << ")" << std::endl;
        std::cout << "(alpha, beta) = (" << alpha << ", " << beta << ")" << std::endl;
        std::cout << "Block sizes: (bm, bn, bk) = (" << bm << ", " << bn << ", " << bk << ")" << std::endl;
        std::cout << "Transpose flags (TA, TB) = (" << ta << ", " << tb << ")" << std::endl;
        std::cout << "Processor grid: (prows, pcols) = (" << p << ", " << q << ")" << std::endl;
    }

    // *******************************
    //   multiply and validate
    // *******************************
    bool ok = test_pdgemm(m, n, k,
                          bm, bn, bk,
                          1, 1, 1,
                          ta, tb,
                          p, q,
                          alpha, beta,
                          rank, MPI_COMM_WORLD);

    int result = ok ? 0 : 1;
    int global_result = 0;

    MPI_Reduce(&result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::string yes_no = global_result == 0 ? "" : " NOT";
        std::cout << "Result is" << yes_no << " CORRECT!" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
