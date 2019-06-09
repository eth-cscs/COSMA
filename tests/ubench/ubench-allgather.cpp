#include <cosma/interval.hpp>
#include <cosma/timer.hpp>

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace cosma;

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
