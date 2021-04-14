#include <costa/grid2grid/memory_utils.hpp>
#include <costa/grid2grid/threads_workspace.hpp>
#include <mkl.h>
#include <chrono>
#include <limits>

int main(int argc, char** argv) {
    int n_rep = 3;
    // dimensions before transposing
    std::vector<int> n_rows = {5000, 10000, 15000, 20000, 25000, 30000}; // 5000;
    std::vector<int> n_cols = {5000, 10000, 15000, 20000, 25000, 30000}; // 10000;

    // not strided
    auto src_stride = n_rows; // 5000;
    auto  dest_stride = n_cols; // 10000;
    bool conjugate = false;

    costa::memory::threads_workspace<double> workspace(256);

    std::vector<long> g2g_times;
    std::vector<long> mkl_times;

    for (int i = 0; i < n_rows.size(); ++i) {
        long g2g_time = std::numeric_limits<long>::max();
        long mkl_time = std::numeric_limits<long>::max();

        src_stride[i] = std::max(n_rows[i], src_stride[i]);
        // since transposed
        dest_stride[i] = std::max(n_cols[i], dest_stride[i]);

        std::vector<double> src(src_stride[i] * n_cols[i]);
        std::vector<double> dest_g2g(dest_stride[i] * n_rows[i]);
        std::vector<double> dest_mkl(dest_stride[i] * n_rows[i]);

        for (int row = 0; row < n_rows[i]; ++row) {
            for (int col = 0; col < n_cols[i]; ++col) {
                src[col * src_stride[i] + row] = col * src_stride[i] + row;
            }
        }

        for (int rep = 0; rep < n_rep; ++rep) {
            // ***********************************
            // transpose with costa 
            // ***********************************
            auto start = std::chrono::steady_clock::now();
            costa::memory::copy_and_transpose<double>(src.data(), n_rows[i], n_cols[i], src_stride[i],
                                                  dest_g2g.data(), dest_stride[i], false, workspace);
            auto end = std::chrono::steady_clock::now();
            g2g_time = std::min(g2g_time, (long) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

            // ***********************************
            // transpose with mkl
            // ***********************************
            start = std::chrono::steady_clock::now();
            mkl_domatcopy('C', 'T', n_rows[i], n_cols[i], 1.0, src.data(), src_stride[i], dest_mkl.data(), dest_stride[i]);
            end = std::chrono::steady_clock::now();
            mkl_time = std::min(mkl_time, (long) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        }

        g2g_times.push_back(g2g_time);
        mkl_times.push_back(mkl_time);

        // ***********************************
        // checking results
        // ***********************************
        int n_rows_t = n_cols[i];
        int n_cols_t = n_rows[i];
        for (int row = 0; row < n_rows_t; ++row) {
            for (int col = 0; col < n_cols_t; ++col) {
                // dest_stride >= n_cols
                auto g2g = dest_g2g[col * dest_stride[i] + row];
                auto mkl = dest_mkl[col * dest_stride[i] + row];
                auto target = src[row * src_stride[i] + col];
                if (g2g != mkl) {
                    std::cout << "Error: (" << col << ", " << row << ") = " << ", g2g = " << g2g << ", mkl = " << mkl << ", target = " << target << std::endl;
                }
            }
        }
    }

    // ***********************************
    // output COSTA timings
    // ***********************************
    std::cout << "COSTA times: " << std::endl;
    for (int i = 0; i < g2g_times.size(); ++i) {
        std::cout << g2g_times[i] << ", ";
    }
    std::cout << std::endl;

    // ***********************************
    // output MKL timings
    // ***********************************
    std::cout << "mkl times: " << std::endl;
    for (int i = 0; i < mkl_times.size(); ++i) {
        std::cout << mkl_times[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}




