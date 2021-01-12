// from std
#include "../utils/pxtran_utils.hpp"
#include <cxxopts.hpp>
#include <unordered_set> 

using namespace costa;

int main(int argc, char **argv) {
    // **************************************
    //   setup command-line parser
    // **************************************
    cxxopts::Options options("COSTA PXTRAN MINIAPP", 
        "A miniapp computing: `C = alpha*A^T + beta*C` and comparing the performance of COSTA (with scalapack wrappers) VS SCALAPACK.");

    // **************************************
    //   readout the command line arguments
    // **************************************
    // matrix dimensions
    // dim(A) = n*m, dim(C) = m*n
    options.add_options()
        ("m,m_dim",
            "number of rows of A and C.", 
            cxxopts::value<int>()->default_value("1000"))
        ("n,n_dim",
            "number of columns of B and C.",
            cxxopts::value<int>()->default_value("1000"))
        ("block_a",
            "block dimensions for matrix A.",
             cxxopts::value<std::vector<int>>()->default_value("128,128"))
        ("block_c",
            "block dimensions for matrix C.",
             cxxopts::value<std::vector<int>>()->default_value("128,128"))
        ("p,p_grid",
            "processor 2D-decomposition.",
             cxxopts::value<std::vector<int>>()->default_value("1,1"))
        ("alpha",
            "alpha parameter in C = alpha*A^T + beta*C",
            cxxopts::value<int>()->default_value("1"))
        ("beta",
            "beta parameter in C = alpha*A^T + beta*C",
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
        ("algorithm", 
            "defines which algorithm (costa, scalapack or both) to run",
            cxxopts::value<std::string>()->default_value("both"))
        ("h,help", "Print usage.")
    ;

    const char** const_argv = (const char**) argv;
    auto result = options.parse(argc, const_argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto m = result["m_dim"].as<int>();
    auto n = result["n_dim"].as<int>();

    auto block_a = result["block_a"].as<std::vector<int>>();
    auto block_c = result["block_c"].as<std::vector<int>>();

    auto p_grid = result["p_grid"].as<std::vector<int>>();

    auto al = result["alpha"].as<int>();
    auto be = result["beta"].as<int>();
    // check if alpha and beta take correct values
    if ((al != 0 && al != 1) || (be != 0 && be != 1)) {
        std::cout << "COSTA (pxtran_miniapp.cpp): ERROR: in this miniapp, \
        --alpha and --beta options can only take values 0 or 1 corresponding to \
        the zero and the unit elements (respectively), of the chosen data type. \
        This is not a requirement of COSTA pxtran wrapper, but just of this miniapp. \
        These elements are chosen because they are well defined also for complex data-types." 
        << std::endl;
        return 0;
    }

    bool test_correctness = result["test"].as<bool>();

    auto n_rep = result["n_rep"].as<int>();

    auto type = result["type"].as<std::string>();
    // transform to lower-case
    std::transform(type.begin(), type.end(), type.begin(), 
        [&](char c) {
            return std::tolower(c);
        }
    );
    // check if the type option takes a correct value
    std::unordered_set<std::string> type_options = {
        "float", "double", "zfloat", "zdouble"
    };
    if (type_options.find(type) == type_options.end()) {
        std::cout << "COSTA (pxtran_miniapp.cpp): ERROR: --type option: can only take the following values: " << std::endl;
        for (const auto& el : type_options) {
            std::cout << el << ", ";
        }
        std::cout << std::endl;
        return 0;
    }

    // make lower-space
    auto algorithm = result["algorithm"].as<std::string>();
    std::transform(algorithm.begin(), algorithm.end(), algorithm.begin(), 
        [&](char c) {
            return std::tolower(c);
        }
    );

    // check if the algorithm option takes a correct value
    std::unordered_set<std::string> algorithm_options = {
        "costa", "scalapack", "both"
    };
    if (algorithm_options.find(algorithm) == algorithm_options.end()) {
        std::cout << "COSTA (pxtran_miniapp.cpp): ERROR: --algorithm option: can only take the following values: " << std::endl;
        for (const auto& el : algorithm_options) {
            std::cout << el << ", ";
        }
        std::cout << std::endl;
        return 0;
    }

    // some basic checks
    if (test_correctness) {
        // if testing correctness, n_rep = 1;
        n_rep = 1;
        std::cout << "COSTA(pxtran_miniapp.cpp): WARNING: correctness checking enabled, setting `n_rep` to 1." << std::endl;
        if (algorithm != "both") {
            std::cout << "COSTA(pxtran_miniapp.cpp): WARNING: correctness checking enabled, setting `algorithm` to `both`." << std::endl;
            algorithm = "both";
        }
    }

    std::vector<long> costa_times;
    std::vector<long> scalapack_times;

    bool result_correct = true;

    // initilize MPI
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    // check if processor grid corresponds to P
    if (p_grid[0] * p_grid[1] != P) {
        p_grid[0] = 1;
        p_grid[1] = P;
        if (rank == 0) {
            std::cout << "COSTA(pxtran_miniapp.cpp): warning: number of processors in the grid must be equal to P, setting grid to 1xP instead." << std::endl;
        }
    }

    // *******************************
    //   perform the multiplication
    // ******************************
    // no blacs functions will be invoked afterwards
    bool exit_blacs = true;
    try {
        if (type == "double") {
            double alpha = double{1.0 * al};
            double beta = double{1.0 * be};
            pxtran_params<double> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PDTRAN on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxtran<double>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times, scalapack_times, 
                                    test_correctness, exit_blacs);
        } else if (type == "float") {
            float alpha = float{1.0f * al};
            float beta = float{1.0f * be};
            pxtran_params<float> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PSTRAN on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxtran<float>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times, scalapack_times,
                                    test_correctness, exit_blacs);

        } else if (type == "zfloat") {
            std::complex<float> alpha = std::complex<float>{1.0f * al};
            std::complex<float> beta = std::complex<float>{1.0f * be};
            pxtran_params<std::complex<float>> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PCTRANU on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxtran<std::complex<float>>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times, scalapack_times,
                                    test_correctness, exit_blacs);
        } else if (type == "zdouble") {
            std::complex<double> alpha = std::complex<double>{1.0 * al};
            std::complex<double> beta = std::complex<double>{1.0 * be};
            pxtran_params<std::complex<double>> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid[0], p_grid[1],
                                         alpha, beta);

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PZTRANU on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxtran<std::complex<double>>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times, scalapack_times,
                                    test_correctness, exit_blacs);
        } else {
            throw std::runtime_error("COSTA(pxtran_miniapp): unknown data type of matrix entries.");
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
        if (algorithm == "both" || algorithm == "costa") {
            std::cout << "COSTA TIMES [ms] = ";
            for (auto &time : costa_times) {
                std::cout << time << " ";
            }
            std::cout << std::endl;
        }

        if (algorithm == "both" || algorithm == "scalapack") {
            std::cout << "SCALAPACK TIMES [ms] = ";
            for (auto &time : scalapack_times) {
                std::cout << time << " ";
            }
            std::cout << std::endl;
        }
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
