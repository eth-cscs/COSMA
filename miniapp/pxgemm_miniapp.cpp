// from std
#include "../utils/pxgemm_utils.hpp"
#include <cxxopts.hpp>

using namespace cosma;

int main(int argc, char **argv) {
    // **************************************
    //   setup MPI and command-line parser
    // **************************************
    cxxopts::Options options("COSMA PXGEMM MINIAPP", 
        "A miniapp computing: `C = alpha*A*B + beta*C` and comparing the performance of COSMA (with scalapack wrappers) VS SCALAPACK.");

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // **************************************
    //   readout the command line arguments
    // **************************************
    // matrix dimensions
    // dim(A) = mxk, dim(B) = kxn, dim(C) = mxn
    options.add_options()
        ("m,m_dim",
            "number of rows of A and C.", 
            cxxopts::value<int>()->default_value("1000"))
        ("n,n_dim",
            "number of columns of B and C.",
            cxxopts::value<int>()->default_value("1000"))
        ("k,k_dim",
            "number of columns of A and rows of B.", 
            cxxopts::value<int>()->default_value("1000"))
        ("block_a",
            "block dimensions for matrix A.",
             cxxopts::value<std::vector<int>>()->default_value("128,128"))
        ("block_b",
            "block dimensions for matrix B.",
             cxxopts::value<std::vector<int>>()->default_value("128,128"))
        ("block_c",
            "block dimensions for matrix C.",
             cxxopts::value<std::vector<int>>()->default_value("128,128"))
        ("p,p_grid",
            "processor 2D-decomposition.",
             cxxopts::value<std::vector<int>>()->default_value("1,1"))
        ("transpose",
            "Transpose/Conjugate flags for A and B.",
             cxxopts::value<std::string>()->default_value("NN"))
        ("alpha",
            "Alpha parameter in C = alpha*A*B + beta*C.",
            cxxopts::value<int>()->default_value("1"))
        ("beta",
            "Beta parameter in C = alpha*A*B + beta*C.",
            cxxopts::value<int>()->default_value("0"))
        ("r,n_rep",
            "number of repetitions",
            cxxopts::value<int>()->default_value("2"))
        ("t,type",
            "data type of matrix entries.",
            cxxopts::value<std::string>()->default_value("double"))
        ("test",
            "test the result correctness.",
            cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage.")
    ;

    auto result = options.parse(argc, argv);

    auto m = result["m_dim"].as<int>();
    auto n = result["n_dim"].as<int>();
    auto k = result["k_dim"].as<int>();

    auto block_a = result["block_a"].as<std::vector<int>>();
    auto block_b = result["block_b"].as<std::vector<int>>();
    auto block_c = result["block_c"].as<std::vector<int>>();

    auto p_grid = result["p_grid"].as<std::vector<int>>();
    if (p_grid[0] * p_grid[0] != P) {
        p_grid[0] = 1;
        p_grid[1] = P;
        if (rank == 0) {
            std::cout << "COSMA(pxgemm_miniapp.cpp): warning: number of processors in the grid must be equal to P, setting grid to 1xP instead." << std::endl;
        }
    }

    auto transpose = result["transpose"].as<std::string>();

    auto al = result["alpha"].as<int>();
    auto be = result["beta"].as<int>();

    bool test_correctness = result["test"].as<bool>();

    auto n_rep = result["n_rep"].as<int>();

    if (test_correctness) {
        // if testing correctness, n_rep = 1;
        n_rep = 1;
        std::cout << "COSMA(pxgemm_miniapp.cpp): warning: correctness checking enabled, setting n_rep to 1." << std::endl;
    }

    auto type = result["type"].as<std::string>();

    char ta = transpose[0];
    char tb = transpose[1];

    std::vector<long> cosma_times(n_rep);
    std::vector<long> scalapack_times(n_rep);

    bool result_correct = true;

    // *******************************
    //   perform the multiplication
    // ******************************
    // no blacs functions will be invoked afterwards
    bool exit_blacs = true;
    try {
        if (type == "double") {
            // create the context here, so that
            // it doesn't have to be created later
            // (this is not necessary)
            auto ctx = cosma::get_context_instance<double>();
            if (rank == 0) {
                ctx->turn_on_output();
            }

            double alpha = double{1.0 * al};
            double beta = double{1.0 * be};
            pxgemm_params<double> params(m, n, k, 
                                         block_a[0], block_a[1],
                                         block_b[0], block_b[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         ta, tb,
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PDGEMM on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemm<double>(params, MPI_COMM_WORLD, n_rep,
                                     cosma_times, scalapack_times, 
                                     test_correctness, exit_blacs);
        } else if (type == "float") {
            // create the context here, so that
            // it doesn't have to be created later
            // (this is not necessary)
            auto ctx = cosma::get_context_instance<float>();
            if (rank == 0) {
                ctx->turn_on_output();
            }

            float alpha = float{1.0f * al};
            float beta = float{1.0f * be};
            pxgemm_params<float> params(m, n, k, 
                                         block_a[0], block_a[1],
                                         block_b[0], block_b[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         ta, tb,
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PSGEMM on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemm<float>(params, MPI_COMM_WORLD, n_rep,
                                    cosma_times, scalapack_times,
                                    test_correctness, exit_blacs);

        } else if (type == "zfloat") {
            // create the context here, so that
            // it doesn't have to be created later
            // (this is not necessary)
            auto ctx = cosma::get_context_instance<std::complex<float>>();
            if (rank == 0) {
                ctx->turn_on_output();
            }

            std::complex<float> alpha = std::complex<float>{1.0f * al};
            std::complex<float> beta = std::complex<float>{1.0f * be};
            pxgemm_params<std::complex<float>> params(m, n, k, 
                                         block_a[0], block_a[1],
                                         block_b[0], block_b[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         ta, tb,
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PCGEMM on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemm<std::complex<float>>(params, MPI_COMM_WORLD, n_rep,
                                     cosma_times, scalapack_times,
                                     test_correctness, exit_blacs);
        } else if (type == "zdouble") {
            // create the context here, so that
            // it doesn't have to be created later
            // (this is not necessary)
            auto ctx = cosma::get_context_instance<std::complex<double>>();
            if (rank == 0) {
                ctx->turn_on_output();
            }

            std::complex<double> alpha = std::complex<double>{1.0 * al};
            std::complex<double> beta = std::complex<double>{1.0 * be};
            pxgemm_params<std::complex<double>> params(m, n, k, 
                                         block_a[0], block_a[1],
                                         block_b[0], block_b[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         ta, tb,
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PZGEMM on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemm<std::complex<double>>(params, MPI_COMM_WORLD, n_rep,
                                     cosma_times, scalapack_times,
                                     test_correctness, exit_blacs);
        } else {
            throw std::runtime_error("COSMA(pxgemm_miniapp): unknown data type of matrix entries.");
        }
    } catch (const std::exception& e) {
        // MPI is already finalized, but just in case
        std::cout << e.what() << std::endl;
        int flag = 0;
        MPI_Finalized(&flag);
        if (!flag) {
            MPI_Abort(MPI_COMM_WORLD, -1);
            MPI_Finalize();
        }
        return 0;
    }

    // *****************
    //   output times
    // *****************
    if (rank == 0) {
        std::cout << "COSMA TIMES [ms] = ";
        for (auto &time : cosma_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;

        std::cout << "SCALAPACK TIMES [ms] = ";
        for (auto &time : scalapack_times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    if (test_correctness) {
        int result = result_correct ? 0 : 1;
        int global_result = 0;
        MPI_Reduce(&result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::string yes_no = global_result == 0 ? "" : " NOT";
            std::cout << "Result is" << yes_no << " CORRECT!" << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
