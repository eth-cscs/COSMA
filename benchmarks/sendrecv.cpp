#include <cosma/interval.hpp>
#include <cosma/timer.hpp>

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace cosma;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int base_size = 1 << 25;
    int local_size = base_size;

    std::vector<double> in(local_size);
    std::vector<double> result(local_size);

    const int n_rep = 10;
    for (int i = 0; i < n_rep; ++i) {
        int target = 1 - rank;
        MPI_Sendrecv(in.data(),
                     local_size,
                     MPI_DOUBLE,
                     target,
                     0,
                     result.data(),
                     local_size,
                     MPI_DOUBLE,
                     target,
                     0,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
    }

    MPI_Finalize();
    return 0;
}
