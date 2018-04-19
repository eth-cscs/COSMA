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
#include <carma.hpp>

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

    //Declare A,B and C CARMA matrices objects
    CarmaMatrix A('A', strategy, rank);
    CarmaMatrix B('B', strategy, rank);
    CarmaMatrix C('C', strategy, rank);

    // fill the matrices with random data
    srand48(rank);
    fillInt(A.matrix());
    fillInt(B.matrix());

    multiply(A, B, C, strategy, MPI_COMM_WORLD);

    output_matrix(A, rank);
    output_matrix(B, rank);
    output_matrix(C, rank);

    MPI_Finalize();

    return 0;
}
