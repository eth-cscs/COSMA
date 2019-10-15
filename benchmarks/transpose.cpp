#include <grid2grid/memory_utils.hpp>
#include <grid2grid/tiling_manager.hpp>
#include <mkl.h>
#include <chrono>

int main(int argc, char** argv) {
    int n_rep = 5;
    // dimensions before transposing
    int n_rows = 5000; // 5000;
    int n_cols = 10000; // 10000;

    int src_stride = n_rows; // 5000;
    int dest_stride = n_cols; // 10000;
    bool conjugate = false;

    src_stride = std::max(n_rows, src_stride);
    // since transposed
    dest_stride = std::max(n_cols, dest_stride);

    std::vector<double> src(src_stride * n_cols);
    std::vector<double> dest_g2g(dest_stride * n_rows);
    std::vector<double> dest_mkl(dest_stride * n_rows);

    std::vector<long> g2g_times(n_rep);
    std::vector<long> mkl_times(n_rep);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            src[j * src_stride + i] = j * src_stride + i;
            // std::cout << src[j*src_stride + i] << ", ";
        }
        // std::cout << std::endl;
    }

    grid2grid::memory::tiling_manager<double> tiling;

    for (int i = 0; i < n_rep; ++i) {
        // ***********************************
        // transpose with grid2grid
        // ***********************************
        auto start = std::chrono::steady_clock::now();
        grid2grid::memory::copy_and_transpose<double>(src.data(), n_rows, n_cols, src_stride,
                                              dest_g2g.data(), dest_stride, false, tiling);
        auto end = std::chrono::steady_clock::now();
        g2g_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // ***********************************
        // transpose with mkl
        // ***********************************
        start = std::chrono::steady_clock::now();
        mkl_domatcopy('C', 'T', n_rows, n_cols, 1.0, src.data(), src_stride, dest_mkl.data(), dest_stride);
        end = std::chrono::steady_clock::now();
        mkl_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    // ***********************************
    // output grid2grid timings
    // ***********************************
    std::sort(g2g_times.begin(), g2g_times.end());
    std::cout << "grid2grid times: " << std::endl;
    for (int i = 0; i < n_rep; ++i) {
        std::cout << g2g_times[i] << ", ";
    }
    std::cout << std::endl;

    // ***********************************
    // output MKL timings
    // ***********************************
    std::sort(mkl_times.begin(), mkl_times.end());
    std::cout << "mkl times: " << std::endl;
    for (int i = 0; i < n_rep; ++i) {
        std::cout << mkl_times[i] << ", ";
    }
    std::cout << std::endl;

    // ***********************************
    // checking results
    // ***********************************
    int n_rows_t = n_cols;
    int n_cols_t = n_rows;
    for (int i = 0; i < n_rows_t; ++i) {
        for (int j = 0; j < n_cols_t; ++j) {
            // dest_stride >= n_cols
            auto g2g = dest_g2g[j * dest_stride + i];
            auto mkl = dest_mkl[j * dest_stride + i];
            auto target = src[i * src_stride + j];
            if (g2g != mkl) {
                std::cout << "Error: (" << j << ", " << i << ") = " << ", g2g = " << g2g << ", mkl = " << mkl << ", target = " << target << std::endl;
            }
        }
    }

    return 0;
}




