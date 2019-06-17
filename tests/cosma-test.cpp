#include <cosma/blas.hpp>

#include <cosma-run.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace cosma;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Strategy strategy(argc, argv);
    auto ctx = cosma::make_context();

    if (rank == 0) {
        std::cout << "Strategy = " << strategy << std::endl;
    }

    // first run without overlapping communication and computation
    bool isOK = run(strategy, ctx, MPI_COMM_WORLD, false);
    MPI_Barrier(MPI_COMM_WORLD);
    // then run with the overlap of communication and computation
    isOK = isOK && run(strategy, ctx, MPI_COMM_WORLD, true);

    MPI_Finalize();

    return rank == 0 ? !isOK : 0;
}
