#include <iostream>
#include <vector>

#include <omp.h>

#include <semiprof/semiprof.hpp>

int main() {
    std::cout << "-----------------------\n"
              << "OpenMP semiprof example\n"
              << "Using " << omp_get_max_threads() << " threads\n"
              << "-----------------------\n\n";


    const size_t n = 1<<28;

    std::vector<double> a(n);
    std::vector<double> b(n);
    std::vector<double> c(n);
    #pragma omp parallel for
    for (auto i=0lu; i<n; ++i) {
        a[i] = 1;
        b[i] = 2;
        c[i] = 3;
    }
    a[n/2] = 0;
    a[n/4] = 2;

    double sum = 0;
    #pragma omp parallel
    {
        PE(add);
        #pragma omp for
        for (auto i=0lu; i<n; ++i) {
            c[i] += a[i] + b[i];
        }
        PL();

        PE(reduce);
        #pragma omp for reduction(+:sum)
        for (auto i=0lu; i<n; ++i) {
            sum += c[i];
        }
        PL();
    }

    std::cout << semiprof::profiler_summary() << "\n\n";

    std::cout << "result : " << sum << "\n";

    return 0;
}
