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

//MPI
#include <mpi.h>

// Local
#include <carma.hpp>

template<typename T>
void fillInt(T& in) {
  std::generate(in.begin(), in.end(),
    [](){ return (int) (10*drand48()); });
}

void output_matrix(CarmaMatrix* M) {
    std::string local = M->which_matrix() + std::to_string(getRank()) + ".txt";
    std::ofstream local_file(local);
    local_file << *M << std::endl;
    local_file.close();
}

int main( int argc, char **argv ) {
    Strategy strategy(argc, argv);
    initCommunication(&argc, &argv);

    int P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    
    if (P != strategy.P) {
        throw(std::runtime_error("Number of processors available not equal \
                                 to the number of processors specified by flag P"));
    }

    if (getRank() == 0) {
        std::cout << strategy << std::endl;
    }

    //Declare A,B and C CARMA matrices objects
    CarmaMatrix* A = new CarmaMatrix('A', strategy, getRank());
    CarmaMatrix* B = new CarmaMatrix('B', strategy, getRank());
    CarmaMatrix* C = new CarmaMatrix('C', strategy, getRank());

    // fill the matrices with random data
    srand48(getRank());
    fillInt(A->matrix());
    fillInt(B->matrix());

    multiply(A, B, C, strategy);

    output_matrix(A);
    output_matrix(B);
    output_matrix(C);

    MPI_Finalize();

    free(A);
    free(B);
    free(C);

    return 0;
}
