// from std
#include "../utils/pxgemr2d_utils.hpp"
#include <cxxopts.hpp>
#include <unordered_set>

using namespace costa;

int main(int argc, char **argv) {
    // **************************************
    //   setup command-line parser
    // **************************************
    cxxopts::Options options("COSTA PXGEMR2D MINIAPP",
        "A miniapp redistributing the matrix between two different block-cyclic distributions and comparing the performance of COSTA VS SCALAPACK.");

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
        ("p,p_grid_a",
            "processor 2D-decomposition.",
             cxxopts::value<std::vector<int>>()->default_value("1,1"))
        ("q,p_grid_c",
            "processor 2D-decomposition.",
             cxxopts::value<std::vector<int>>()->default_value("1,1"))
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

    auto p_grid_a = result["p_grid_a"].as<std::vector<int>>();
    auto p_grid_c = result["p_grid_c"].as<std::vector<int>>();

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
        std::cout << "COSTA (pxgemr2d_miniapp.cpp): ERROR: --type option: can only take the following values: " << std::endl;
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
        std::cout << "COSTA (pxgemr2d_miniapp.cpp): ERROR: --algorithm option: can only take the following values: " << std::endl;
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
        std::cout << "COSTA(pxgemr2d_miniapp.cpp): WARNING: correctness checking enabled, setting `n_rep` to 1." << std::endl;
        if (algorithm != "both") {
            std::cout << "COSTA(pxgemr2d_miniapp.cpp): WARNING: correctness checking enabled, setting `algorithm` to `both`." << std::endl;
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

    int Pa = p_grid_a[0] * p_grid_a[1];
    int Pc = p_grid_c[0] * p_grid_c[1];
    int num_ranks = std::max(Pa, Pc);

    // check if processor grid corresponds to P
    if (num_ranks != P) {
        p_grid_a[0] = 1;
        p_grid_a[1] = P;
        p_grid_c[0] = 1;
        p_grid_c[1] = P;
        if (rank == 0) {
            std::cout << "COSTA(pxgemr2d_miniapp.cpp): warning: number of processors in the grid must be equal to P, setting grid to 1xP instead." << std::endl;
        }
    }

    // *******************************
    //   perform the reshuffling
    // ******************************
    // no blacs functions will be invoked afterwards
    bool exit_blacs = true;
    try {
        if (type == "double") {
            pxgemr2d_params<double> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid_a[0], p_grid_a[1],
                                         p_grid_c[0], p_grid_c[1]
                                         );

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PDGEMR2D on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemr2d<double>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times,
                                    scalapack_times,
                                    test_correctness, exit_blacs);
        } else if (type == "float") {
            pxgemr2d_params<float> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid_a[0], p_grid_a[1],
                                         p_grid_c[0], p_grid_c[1]
                    );

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PSGEMR2D on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemr2d<float>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times,
                                    scalapack_times,
                                    test_correctness, exit_blacs);

        } else if (type == "zfloat") {
            pxgemr2d_params<std::complex<float>> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid_a[0], p_grid_a[1],
                                         p_grid_c[0], p_grid_c[1]
                    );

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PCGEMR2D on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemr2d<std::complex<float>>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times,
                                    scalapack_times,
                                    test_correctness, exit_blacs);
        } else if (type == "zdouble") {
            pxgemr2d_params<std::complex<double>> params(m, n,
                                         block_a[0], block_a[1],
                                         block_c[0], block_c[1],
                                         p_grid_a[0], p_grid_a[1],
                                         p_grid_c[0], p_grid_c[1]
                    );

            // **************************************
            //    output the problem description
            // **************************************
            if (rank == 0) {
                std::cout << "Running PZGEMR2D on the following problem:" << std::endl;
                std::cout << params << std::endl;
            }

            result_correct = benchmark_pxgemr2d<std::complex<double>>(params, MPI_COMM_WORLD, n_rep,
                                    algorithm,
                                    costa_times,
                                    scalapack_times,
                                    test_correctness, exit_blacs);
        } else {
            throw std::runtime_error("COSTA(pxgemr2d_miniapp): unknown data type of matrix entries.");
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
