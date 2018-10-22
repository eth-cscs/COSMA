#pragma once
#include <chrono>
#include <mpi.h>
#include <string>
#include <iostream>

class Timer {
public:
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    int n_rep_;
    std::string region;
    MPI_Comm comm_;
    time_point start;

    Timer(int n_rep, std::string reg = "", MPI_Comm comm = MPI_COMM_WORLD) : n_rep_(n_rep), region(reg), comm_(comm) {
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
