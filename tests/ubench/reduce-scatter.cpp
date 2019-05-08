#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <interval.hpp>
#include <timer.hpp>

using namespace cosma;

int main( int argc, char **argv ) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int base_size = 1 << 25;
    int local_size = base_size;

    std::vector<double> in(local_size);
    std::vector<double> result(local_size/2);
    std::vector<int> recv_counts = {local_size/2, local_size/2};
    std::vector<int> dspls = {0, local_size/2};

    const int n_rep = 10;
    for (int i = 0; i < n_rep; ++i) {
        int target = 1 - rank;
        MPI_Reduce_scatter(in.data(), result.data(), recv_counts.data(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
