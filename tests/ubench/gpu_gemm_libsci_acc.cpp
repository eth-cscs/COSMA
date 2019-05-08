#include <chrono>
#include <libsci_acc.h>
#include <vector>
#include <iostream>

long libsci_acc_dgemm(int m, int n, int k) {
    double* a, *b, *c;

    double alpha = 1.0;
    double beta = 0.0;

    libsci_acc_HostAlloc((void**)&a, sizeof(double)*m*k);
    libsci_acc_HostAlloc((void**)&b, sizeof(double)*k*n);
    libsci_acc_HostAlloc((void**)&c, sizeof(double)*m*n);

    // perform dgemm
    auto start = std::chrono::steady_clock::now();
    dgemm('n', 'n', m, n, k, alpha, a, m, b, k, beta, c, m);
    auto end = std::chrono::steady_clock::now();

    libsci_acc_HostFree(a);
    libsci_acc_HostFree(b);
    libsci_acc_HostFree(c);

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main(int argc, char* argv[]) {
    // initialization
    libsci_acc_init();

    // std::vector<int> dims = {500, 1000, 2000, 4000, 8000, 16000, 32000};
    std::vector<int> dims = {32000};
    int n_iter = 1;

    for (const int& dim : dims) {
        std::cout << "Dimension = " << dim << std::endl;
        double t_avg_libsci = 0;

        for (int i = 0; i < n_iter+1; ++i) {
            long t_libsci = libsci_acc_dgemm(dim, dim, dim);

            if (i == 0) continue;

            t_avg_libsci += t_libsci;
        }
        std::cout << "libsci average time [ms]: " << 1.0*t_avg_libsci/n_iter << std::endl;
    }
    libsci_acc_finalize();
}
