// from std
#include <costa/layout.hpp>
#include <costa/grid2grid/transform.hpp>
#include <costa/grid2grid/cantor_mapping.hpp>

#include <cxxopts.hpp>
#include <unordered_set> 

using namespace costa;

int main(int argc, char **argv) {
    // **************************************
    //   setup command-line parser
    // **************************************
    cxxopts::Options options("COSTA BLOCK-CYCLIC MINIAPP", 
        "A miniapp computing: `C = alpha*op(A) + beta*C` where op = none, transpose or conjugate.");

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

    MPI_Comm comm = MPI_COMM_WORLD;

    // initilize MPI
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // check if processor grid corresponds to P
    if (p_grid[0] * p_grid[1] != P) {
        p_grid[0] = 1;
        p_grid[1] = P;
        if (rank == 0) {
            std::cout << "COSTA(block_cyclic.cpp): warning: number of processors in the grid must be equal to P, setting grid to 1xP instead." << std::endl;
        }
    }

    std::vector<double> a(m*n);
    auto init_layout = costa::block_cyclic_layout<double>(
            m, n, // global matrix dimension
            block_a[0], block_a[1], // block sizes
            1, 1, // submatrix start 
                  // (1-based, because of scalapack.
                  // Since we take full matrix, it's 1,1)
            m, n, // submatrix size (since full matrix, it's mat dims)
            p_grid[0], p_grid[1], // processor grid
            'R', // processor grid ordering row-major
            0, 0, // coords or ranks oweing the first row (0-based)
            &a[0], // local data of full matrix
            m, // local leading dimension
            rank // current rank
    );

    std::vector<double> b(n*m);
    auto final_layout = costa::block_cyclic_layout<double>(
            n, m, // global matrix dimension
            block_c[0], block_c[1], // block sizes
            1, 1, // submatrix start 
                  // (1-based, because of scalapack.
                  // Since we take full matrix, it's 1,1)
            n, m, // submatrix size (since full matrix, it's mat dims)
            p_grid[0], p_grid[1], // processor grid
            'R', // processor grid ordering row-major
            0, 0, // coords or ranks oweing the first row (0-based)
            &b[0], // local data of full matrix
            n, // local leading dimension
            rank // current rank
    );

    auto f = [](int i, int j) -> double {
        return static_cast<double>(costa::cantor_pairing(i, j)); 
    };
    init_layout.initialize(f);

    auto transposed_f = [](int i, int j) -> double {
        return static_cast<double>(costa::cantor_pairing(j, i));
    };

    costa::transform(init_layout, final_layout, 'T', 1.0, 0.0, comm);

    bool ok = final_layout.validate(transposed_f, 1e-6);

    // collect all results and check the correctness
    int res = ok ? 0 : 1;
    int global_result = 0;
    MPI_Reduce(&res, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::string yes_no = global_result == 0 ? "" : " NOT";
        std::cout << "Result is" << yes_no << " CORRECT!" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
