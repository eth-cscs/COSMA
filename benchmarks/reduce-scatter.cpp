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

    const int n_rep = 2;
    int scaling_factor = P;

    /*
    for (int i = -10; i <= 10; ++i) {
        int dim = (scaling_factor+i)*P;
        int block_size = (dim/P) * dim;
        int total_size = block_size * P;

        if (rank == 0)
            std::cout << "dim = " << dim << std::endl;

        std::vector<double> in(total_size);
        std::vector<double> result(block_size);
        MPI_Request reqs[2];

        {
            Timer time(n_rep, "MPI_Reduce_scatter_block");
            for (int i = 0; i < n_rep; ++i) {
                MPI_Ireduce_scatter_block(in.data(),
                                   result.data(),
                                   block_size/2,
                                   MPI_DOUBLE,
                                   MPI_SUM,
                                   MPI_COMM_WORLD,
                                   &reqs[0]);
                MPI_Ireduce_scatter_block(in.data(),
                                   result.data(),
                                   block_size/2,
                                   MPI_DOUBLE,
                                   MPI_SUM,
                                   MPI_COMM_WORLD,
                                   &reqs[1]);
                MPI_Waitall(2, &reqs[0], MPI_STATUSES_IGNORE);
            }
        }
    }
    */
    int dim = 17408;
    int block_size = (dim/P) * dim;
    int total_size = block_size * P;
    std::vector<double> in(total_size);
    std::vector<double> result(block_size);
    {
        Timer time(n_rep, "MPI_Reduce_scatter_block");
        for (int i = 0; i < n_rep; ++i) {
            MPI_Reduce_scatter_block(in.data(),
                               result.data(),
                               block_size,
                               MPI_DOUBLE,
                               MPI_SUM,
                               MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
