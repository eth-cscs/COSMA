#include <cosma/blas.hpp>

#include <mpi.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unistd.h>
#include <vector>

class Timer {
  public:
    using time_point =
        std::chrono::time_point<std::chrono::high_resolution_clock>;

    int n_rep_;
    std::string region;
    MPI_Comm comm_;
    time_point start;

    Timer(int n_rep, std::string reg = "", MPI_Comm comm = MPI_COMM_WORLD)
        : n_rep_(n_rep)
        , region(reg)
        , comm_(comm) {
        MPI_Barrier(comm);
        start = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        auto time =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                .count();
        long long max_time, min_time, sum_time;
        MPI_Reduce(&time, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, comm_);
        MPI_Reduce(&time, &min_time, 1, MPI_LONG_LONG, MPI_MIN, 0, comm_);
        MPI_Reduce(&time, &sum_time, 1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        int rank, size;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &size);
        if (rank == 0) {
            std::cout << region << " MIN TIME [ms]: " << 1.0 * min_time / n_rep_
                      << std::endl;
            std::cout << region << " MAX TIME [ms]: " << 1.0 * max_time / n_rep_
                      << std::endl;
            std::cout << region
                      << " AVG TIME [ms]: " << 1.0 * sum_time / (n_rep_ * size)
                      << std::endl;
            std::cout << "\n";
        }
    }
};

std::pair<int, int> group_and_offset(int P, int divisor, int rank) {
    int subset_size = P / divisor;
    int subint_index = rank / subset_size;
    int offset = rank - subint_index * subset_size;
    return {subint_index, offset};
}

void solve(double *A, double *B, double *C, int m, int n, int k) {
    // multiply square matrices with dimensions sqrt(local_size)
    cosma::dgemm(m, n, k, 1.0, A, m, B, k, 0.0, C, m);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int divisor = 2;
    int m = 5000;
    int k = 2000;
    int n = 2000;
    int n_iter = 3;
    size_t local_size = k * n / divisor;
    float waiting_time = 0.7f;
    std::vector<double> local_buffer(local_size);
    std::vector<double> global_buffer(local_size * divisor);

    std::vector<double> a(m / divisor * k);
    std::vector<double> b(k * n);
    std::vector<double> c(m / divisor * n);

    // initialize dgemm
    for (int i = 0; i < 5; ++i) {
        solve(a.data(), b.data(), c.data(), m / divisor, n / divisor, k);
    }

    {
        Timer dgemm_small(10, "dgemm subproblem");
        for (int i = 0; i < 10; ++i) {
            solve(a.data(), b.data(), c.data(), m / divisor, n / divisor, k);
        }
    }
    {
        Timer dgemm_large(10, "dgemm large problem");
        for (int i = 0; i < 10; ++i) {
            solve(a.data(), b.data(), c.data(), m / divisor, n, k);
        }
    }

    int gp, off;
    std::tie(gp, off) = group_and_offset(P, divisor, rank);
    MPI_Comm subcom;
    MPI_Comm_split(MPI_COMM_WORLD, off, gp, &subcom);

    MPI_Request req[2 * (divisor - 1)];

    int reqi = 0;
    for (int i = 0; i < divisor; ++i) {
        if (i != gp) {
            int offset = i * local_size;

            MPI_Recv_init(global_buffer.data() + offset,
                          local_size,
                          MPI_DOUBLE,
                          i,
                          0,
                          subcom,
                          &req[reqi]);
            MPI_Send_init(local_buffer.data(),
                          local_size,
                          MPI_DOUBLE,
                          i,
                          0,
                          subcom,
                          &req[divisor - 1 + reqi]);
            reqi++;
        }
    }

    {
        Timer timer_async(1, "asynchronous");
        MPI_Startall(2 * (divisor - 1), req);

        // do the work
        solve(a.data(), b.data(), c.data(), m / divisor, n / divisor, k);
        // usleep(waiting_time * 1e6);

        for (int i = 0; i < divisor - 1; ++i) {
            int idx = -1;
            MPI_Waitany(divisor - 1, req, &idx, MPI_STATUS_IGNORE);
            // if (idx >= rank) idx++;
            solve(a.data(), b.data(), c.data(), m / divisor, n / divisor, k);
            // usleep(waiting_time * 1e6);
        }

        MPI_Waitall(divisor - 1, req + divisor - 1, MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    {
        Timer timer_sync(1, "synchronous");

        MPI_Allgather(local_buffer.data(),
                      local_size,
                      MPI_DOUBLE,
                      global_buffer.data(),
                      local_size,
                      MPI_DOUBLE,
                      subcom);

        solve(a.data(), b.data(), c.data(), m / divisor, n, k);
        // usleep(1e6 * divisor * waiting_time);
    }

    MPI_Comm_free(&subcom);

    MPI_Finalize();
}
