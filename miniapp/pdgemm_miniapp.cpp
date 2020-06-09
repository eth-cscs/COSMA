// from std
#include "../utils/pxgemm_utils.hpp"
#include "../utils/parse_strategy.hpp"

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

    // create the context here, so that
    // it doesn't have to be created later
    // (this is not necessary)
    auto ctx = cosma::get_context_instance<double>();
    if (rank == 0) {
        ctx->turn_on_output();
    }

    // **************************************
    //   readout the command line arguments
    // **************************************
    // matrix dimensions
    // dim(A) = mxk, dim(B) = kxn, dim(C) = mxn
    auto m = options::next_int("-m", "--m_dim", "number of rows of A and C.", 1000);
    auto n = options::next_int("-n", "--n_dim", "number of columns of B and C.", 1000);
    auto k = options::next_int("-k", "--k_dim", "number of columns of A and rows of B.", 1000);

    // block sizes
    auto block_a = options::next_int_pair("-ba", "--block_a", "block size for the number of rows of A.", 128);
    auto block_b = options::next_int_pair("-bb", "--block_b", "block size for the number of rows of B.", 128);
    auto block_c = options::next_int_pair("-bc", "--block_c", "block size for the number of rows of C.", 128);

    // indices of submatrices of A, B and C to be multiplied
    // if 1 then full matrices A and B are multiplied
    int submatrix_m = 1;
    int submatrix_n = 1;
    int submatrix_k = 1;

    // processor grid decomposition
    auto p = options::next_int("-p", "--p_row", "number of rows in a processor grid.", 1);
    auto q = options::next_int("-q", "--q_row", "number of columns in a processor grid.", P);

    // alpha and beta of multiplication
    auto alpha = options::next_double("-a", "--alpha", "Alpha parameter in C = alpha*A*B + beta*C", 1.0);
    auto beta = options::next_double("-b", "--beta", "Beta parameter in C = alpha*A*B + beta*C", 0.0);

    // number of repetitions
    auto n_rep = options::next_int("-r", "--n_rep", "number of repetitions", 2);

    // transpose flags
    bool trans_a = options::flag_exists("-ta", "--trans_a");
    bool trans_b = options::flag_exists("-tb", "--trans_b");

    char ta = trans_a ? 'T' : 'N';
    char tb = trans_b ? 'T' : 'N';

    if (p * q != P) {
        std::runtime_error("Number of processors in a grid has to match the number of available ranks.");
    }

    pxgemm_params<double> params(m, n, k, 
                                 block_a.first, block_a.second,
                                 block_b.first, block_b.second,
                                 block_c.first, block_c.second,
                                 p, q,
                                 ta, tb,
                                 alpha, beta);

    // **************************************
    //    output the problem description
    // **************************************
    if (rank == 0) {
        std::cout << "Running PDGEMM on the following problem:" << std::endl;
        std::cout << params << std::endl;
    }

    std::vector<long> cosma_times(n_rep);
    std::vector<long> scalapack_times(n_rep);

    // *******************************
    //   perform the multiplication
    // ******************************
    // no blacs functions will be invoked afterwards
    bool exit_blacs = true;
    try {
        benchmark_pxgemm<double>(params, MPI_COMM_WORLD, n_rep,
                               cosma_times, scalapack_times, exit_blacs);
    } catch (const std::exception& e) {
        // MPI is already finalized, but just in case
        int flag = 0;
        MPI_Finalized(&flag);
        if (!flag) {
            MPI_Finalize();
        }
        return 0;
    }

    // *****************
    //   output times
    // *****************
    if (rank == 0) {
        std::cout << "COSMA PDGEMM TIMES [ms] = ";
        for (auto &time : cosma_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;

        std::cout << "SCALAPACK PDGEMM TIMES [ms] = ";
        for (auto &time : scalapack_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
