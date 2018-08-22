// STL
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

//Blas
#include "blas.h"

// Local
#include <multiply.hpp>

template<typename T>
void fillInt(T& in) {
  std::generate(in.begin(), in.end(),
    [](){ return (int) (10*drand48()); });
}

void output_matrix(CosmaMatrix& M, int rank) {
    std::string local = M.which_matrix() + std::to_string(rank) + ".txt";
    std::ofstream local_file(local);
    local_file << M << std::endl;
    local_file.close();
}

int run(Strategy& s, MPI_Comm comm=MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //Declare A,B and C CARMA matrices objects
    CosmaMatrix A('A', s, rank);
    CosmaMatrix B('B', s, rank);
    CosmaMatrix C('C', s, rank);

    // fill the matrices with random data
    srand48(rank);
    fillInt(A.matrix());
    fillInt(B.matrix());

    MPI_Barrier(comm);
    multiply(A, B, C, s, comm, s.one_sided_communication);

    output_matrix(A, rank);
    output_matrix(B, rank);
    output_matrix(C, rank);
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

    run(strategy, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
