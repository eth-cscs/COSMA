#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../utils/parse_strategy.hpp"
#include "../utils/cosma_utils.hpp"

using namespace cosma;
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Strategy strategy = (Strategy) parse_strategy(argc, argv);
    auto ctx = cosma::make_context<double>();

    if (rank == 0) {
        std::cout << "Strategy = " << strategy << std::endl;
    }

    // first run without overlapping communication and computation
    bool isOK = test_cosma<double>(strategy, ctx, MPI_COMM_WORLD, false);
    MPI_Barrier(MPI_COMM_WORLD);
    // then run with the overlap of communication and computation
    isOK = isOK && test_cosma<double>(strategy, ctx, MPI_COMM_WORLD, true);

    MPI_Finalize();

    return rank == 0 ? !isOK : 0;
}
