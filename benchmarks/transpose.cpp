#include <grid2grid/memory_utils.hpp>
#include <mkl.h>
#include <chrono>

int main(int argc, char** argv) {
    int n_rep = 5;
    // dimensions before transposing
    int n_rows = 5000;
    int n_cols = 1000;

    int src_stride = 5000;
    int dest_stride = 1000;
    bool conjugate = false;

    src_stride = std::max(n_rows, src_stride);
    // since transposed
    dest_stride = std::max(n_cols, dest_stride);

    std::vector<double> src(src_stride * n_cols);
    std::vector<double> dest(dest_stride * n_rows);

    std::vector<long> g2g_times(n_rep);
    std::vector<long> mkl_times(n_rep);

    for (int i = 0; i < n_rep; ++i) {
        // ***********************************
        // transpose with grid2grid
        // ***********************************
        auto start = std::chrono::steady_clock::now();
        grid2grid::memory::copy_and_transpose(src.data(), n_rows, n_cols, src_stride,
                                              dest.data(), dest_stride, false);
        auto end = std::chrono::steady_clock::now();
        g2g_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // ***********************************
        // transpose with mkl
        // ***********************************
        start = std::chrono::steady_clock::now();
        mkl_domatcopy('C', 'T', n_rows, n_cols, 1.0, src.data(), src_stride, dest.data(), dest_stride);
        end = std::chrono::steady_clock::now();
        mkl_times[i] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    // ***********************************
    // output grid2grid results
    // ***********************************
    std::sort(g2g_times.begin(), g2g_times.end());
    std::cout << "grid2grid times: " << std::endl;
    for (int i = 0; i < n_rep; ++i) {
        std::cout << g2g_times[i] << ", ";
    }
    std::cout << std::endl;

    // ***********************************
    // output MKL results
    // ***********************************
    std::sort(g2g_times.begin(), g2g_times.end());
    std::cout << "mkl times: " << std::endl;
    for (int i = 0; i < n_rep; ++i) {
        std::cout << mkl_times[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}




