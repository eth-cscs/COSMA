#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <interval.hpp>

class Timer {
public:
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    MPI_Comm comm_;
    time_point start;
    std::string region;
    int n_rep_;

    Timer(int n_rep, std::string reg = "", MPI_Comm comm = MPI_COMM_WORLD) : n_rep_(n_rep), comm_(comm), region(reg) {
        MPI_Barrier(comm);
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        long long max_time, min_time, sum_time;
        MPI_Reduce(&time, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, comm_);
        MPI_Reduce(&time, &min_time, 1, MPI_LONG_LONG, MPI_MIN, 0, comm_);
        MPI_Reduce(&time, &sum_time, 1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        int rank, size;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &size);
        if (rank == 0) {
            std::cout << region << " MIN TIME [ms]: " << 1.0*min_time / n_rep_ << std::endl;
            std::cout << region << " MAX TIME [ms]: " << 1.0*max_time / n_rep_ << std::endl;
            std::cout << region << " AVG TIME [ms]: " << 1.0*sum_time / (n_rep_ * size) << std::endl;
            std::cout << "\n";

        }
    }
};

int main( int argc, char **argv ) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int base_size = 1500000;
    int var = base_size / 10;
    int local_size = base_size + ((rank % 2 == 0) ? var : 0);
    int max_size = -1;
    int total_size = 0;

    std::vector<int> sizes(P);
    std::vector<int> dspls(P);

    for (int i = 0; i < P; ++i) {
        int local_size = base_size + ((i % 2 == 0) ? var : 0);
        max_size = std::max(max_size, local_size);
        sizes[i] = local_size;
        dspls[i] = total_size;
        total_size += local_size;
    }

    std::vector<double> in(local_size);
    std::vector<double> in_padded(max_size);

    std::vector<double> result(total_size);
    std::vector<double> result_padded(P*max_size);

    const int n_rep = 30;

    {
        Timer time(n_rep, "MPI_Allgatherv");
        for (int i = 0; i < n_rep; ++i) {
            MPI_Allgatherv(in.data(), local_size, MPI_DOUBLE, result.data(),
                    sizes.data(), dspls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }

    {
        Timer time(n_rep, "MPI_Allgather");
        for (int i = 0; i < n_rep; ++i) {
            MPI_Allgather(in_padded.data(), max_size, MPI_DOUBLE, result_padded.data(),
                    max_size, MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
