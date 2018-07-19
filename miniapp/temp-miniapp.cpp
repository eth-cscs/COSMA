// STL
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>

//Blas
#include "blas.h"

// Local
#include <multiply.hpp>

template<typename T>
void fillInt(T& in) {
  std::generate(in.begin(), in.end(),
    [](){ return (int) (10*drand48()); });
}

void output_matrix(CarmaMatrix& M, int rank) {
    std::string local = M.which_matrix() + std::to_string(rank) + ".txt";
    std::ofstream local_file(local);
    local_file << M << std::endl;
    local_file.close();
}

long run(Strategy& s, MPI_Comm comm=MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //Declare A,B and C CARMA matrices objects
    CarmaMatrix A('A', s, rank);
    CarmaMatrix B('B', s, rank);
    CarmaMatrix C('C', s, rank);

    // fill the matrices with random data
    srand48(rank);
    fillInt(A.matrix());
    fillInt(B.matrix());

    MPI_Barrier(comm);
    auto start = std::chrono::steady_clock::now();
    multiply(A, B, C, s, comm, s.one_sided_communication);
    MPI_Barrier(comm);
    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main( int argc, char **argv ) {
    Strategy strategy(argc, argv);
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (P != strategy.P) {
        throw(std::runtime_error("Number of processors available not equal \
                                 to the number of processors specified by flag P"));
    }

    if (rank == 0) {
        std::cout << strategy << std::endl;
    }

    int n_iter = 5;
    long time = std::numeric_limits<long>::max();
    for (int i = 0; i < n_iter+1; ++i) {
        long t_run = run(strategy, MPI_COMM_WORLD);
        if (i == 0) continue;
        time = std::min(time, t_run);
    }

    if (rank == 0) {
        std::cout << "CARMA MIN TIME [ms] = " << time << std::endl;
    }

    MPI_Finalize();

    return 0;
}
