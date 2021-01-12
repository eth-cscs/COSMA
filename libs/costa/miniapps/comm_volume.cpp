// from std

#include <costa/grid2grid/grid2D.hpp>
#include <costa/grid2grid/ranks_reordering.hpp>
#include <costa/grid2grid/transform.hpp>
#include <cxxopts.hpp>
#include <random>
#include <vector>

using namespace costa;

std::vector<int> split(int n, int block) {
    std::vector<int> splits = {0};
    int tick = 0;
    while (n - tick > 0) {
        int new_tick = std::min(tick + block, n);
        splits.push_back(new_tick);
        tick = new_tick;
    }
    return splits;
}

int main(int argc, char **argv) {
    // **************************************
    //   setup command-line parser
    // **************************************
    cxxopts::Options options("COSTA COMM-VOLUME MINIAPP",
        "A miniapp computing the communication volume for: `C = alpha*op(A) + beta*C` where op = none, transpose or conjugate with and without process relabeling");

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
            "block dimensions for matrix A.",
             cxxopts::value<std::vector<int>>()->default_value("-1,-1"))
        ("p,p_grid_a",
            "processor 2D-decomposition.",
             cxxopts::value<std::vector<int>>()->default_value("1,1"))
        ("q,p_grid_c",
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

    auto p_grid_a = result["p_grid_a"].as<std::vector<int>>();
    auto p_grid_c = result["p_grid_c"].as<std::vector<int>>();

    int P = std::max(p_grid_a[0] * p_grid_a[1], p_grid_c[0] * p_grid_c[1]);

    if (block_c[0] == -1 || block_c[1] == -1) {
        block_c[0] = m/p_grid_c[0];
        block_c[1] = n/p_grid_c[1];
    }
    // std::vector<int> block_c_col = {n/p_grid[1], m/p_grid[0]};
    // block_c = block_c_col;
    std::cout << "block = " << block_c[0] << ", " << block_c[1] << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, P-1);

    // init grid
    std::vector<int> row_split_a = split(m, block_a[0]);
    std::vector<int> col_split_a = split(n, block_a[1]);
    int n_blocks_row_a = row_split_a.size() - 1;
    int n_blocks_col_a = col_split_a.size() - 1;

    std::vector<std::vector<int>> owners_a(n_blocks_row_a,
                                           std::vector<int>(n_blocks_col_a));

    // std::cout << "A layout : " << std::endl;
    // row-major process ordering
    for (int i = 0; i < n_blocks_row_a; ++i) {
        for (int j = 0; j < n_blocks_col_a; ++j) {
            int p_row = i % p_grid_a[0];
            int p_col = j % p_grid_a[1];
            int p = p_row * p_grid_a[1] + p_col;
            // std::cout << "(" << i << ", " << j << ") -> rank " << p << "(" << p_row << ", "  << p_col << ")"<< std::endl;
            // owners_a[i][j] = dist(gen);
            owners_a[i][j] = p;
        }
    }

    assigned_grid2D init_grid{{std::move(row_split_a),
                               std::move(col_split_a)},
                               std::move(owners_a), P};

    // target grid
    std::vector<int> row_split_c = split(m, block_c[0]);
    std::vector<int> col_split_c = split(n, block_c[1]);
    int n_blocks_row_c = row_split_c.size() - 1;
    int n_blocks_col_c = col_split_c.size() - 1;

    std::vector<std::vector<int>> owners_c(n_blocks_row_c,
                                         std::vector<int>(n_blocks_col_c));

    // column-major process ordering
    for (int i = 0; i < n_blocks_row_c; ++i) {
        for (int j = 0; j < n_blocks_col_c; ++j) {
            int p_row = i % p_grid_c[0];
            int p_col = j % p_grid_c[1];
            int p = p_col * p_grid_c[0] + p_row;
            // std::cout << "(" << i << ", " << j << ") -> rank " << p << "(" << p_row << ", "  << p_col << ")"<< std::endl;
            owners_c[i][j] = p;
        }
    }

    assigned_grid2D target_grid{{std::move(row_split_c),
                               std::move(col_split_c)},
                               std::move(owners_c), P};

    bool reordered = false;
    auto comm_vol = costa::communication_volume(init_grid, target_grid);
    std::vector<int> rank_permutation = costa::optimal_reordering(comm_vol, P, reordered);
    target_grid.reorder_ranks(rank_permutation);
    for (int i = 0; i < P; ++i) {
        // if (rank_permutation[i] != i)
            // std::cout << i << "->" << rank_permutation[i] << std::endl;
    }

    // percent of communication volume reduction
    auto new_comm_vol = costa::communication_volume(init_grid, target_grid);

    auto comm_vol_total = comm_vol.total_volume();
    auto new_comm_vol_total = new_comm_vol.total_volume();

    auto diff = (long long) comm_vol_total - (long long) new_comm_vol_total;
    if (comm_vol_total > 0) {
        std::cout << "Comm volume reduction [%] = " << 100.0 * diff / comm_vol_total << std::endl;
    } else {
        std::cout << "Initial comm vol = 0, nothing to improve." << std::endl;
    }

    return 0;
}
