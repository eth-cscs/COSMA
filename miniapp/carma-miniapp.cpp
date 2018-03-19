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

// CMD parser
#include <cmd_parser.h>

// Local
#include <communication.h>
#include <carma.h>
#include <matrix.hpp>

// profiler
#include <profiler.h>

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
    initCommunication(&argc, &argv);
    auto m = read_int(argc, argv, "-m", 4096);
    auto n = read_int(argc, argv, "-n", 4096);
    auto k = read_int(argc, argv, "-k", 4096);
    auto r = read_int(argc, argv, "-r", 4);
    auto char* patt = read_string(argc, argv, "-p", "bbbb");
    std::string pattern(patt);
    auto char* divPatternStr = read_string(argc, argv, "-d", "211211211211");
    std::string divPatternString(divPatternStr);

    //use only lower case
    std::transform(pattern.begin(), pattern.end(), pattern.begin(),
        [](char c) { return std::tolower(c); });
    if ( pattern.size() != r || !std::all_of(pattern.cbegin(),pattern.cend(),
        [](char c) {
          return (c=='b')||(c=='d');
         })) {
           std::cout << "Recursive pattern " << pattern << " malformed expression\n";
           exit(-1);
    }

    std::vector<int> divPattern;
    auto it = divPatternString.cbegin();

    for (int i=0; i<r; i++) {
        bool isNonZero=true;
        for(int j=0; j<3; j++ ) {
            if (it != divPatternString.cend()) {
                int val=std::stoi(std::string(it,it+1));
                divPattern.push_back(val);
                isNonZero &= (val!=0);
            } else {
              std::cout << "Recursive division pattern " << divPatternString << " has wrong size\n";
              exit(-1);
            }
            it++;
        }
        if (!isNonZero){
          std::cout << "Recursive division pattern " << divPatternString << "contains 3 zeros in a step\n";
          exit(-1);
        }
    }

    bool isOK;
    int P;
    MPI_Comm_size( MPI_COMM_WORLD, &P );

    // Check if the parameters make sense.
    int prodBFS = 1;

    for (int i = 0; i < r; i++) {
        int divM = divPattern[3*i];
        int divN = divPattern[3*i + 1];
        int divK = divPattern[3*i + 2];

        if (pattern[i] == 'b') {
          prodBFS *= divM * divN * divK;
        }
    }

    // Check if we are using too few processors!
    if (P < prodBFS) {
      std::cout << "Too few processors available for the given steps. The number of processors should be at least " << std::to_string(prodBFS) << ". Aborting the application.\n";
        exit(-1);
    }

    if( getRank() == 0 ) {
        std::cout<<"Benchmarking "<<m<<"*"<<n<<"*"<<k<<" multiplication using "
            <<P<<" processes"<<std::endl;
        std::cout<<"Division pattern is: "<<pattern<<" - "
            <<divPatternString<<std::endl;
    }

    //Declare A,B and C CARMA matrices objects
    CarmaMatrix* A = new CarmaMatrix('A', m, k, P, r, pattern.cbegin(), divPattern.cbegin(), getRank());
    CarmaMatrix* B = new CarmaMatrix('B', k, n, P, r, pattern.cbegin(), divPattern.cbegin(), getRank());
    CarmaMatrix* C = new CarmaMatrix('C', m, n, P, r, pattern.cbegin(), divPattern.cbegin(), getRank());

    // initial sizes
    auto sizeA=A->initial_size();
    auto sizeB=B->initial_size();
    auto sizeC=C->initial_size();

    // fill the matrices with random data
    srand48(getRank());
    fillInt(A->matrix());
    fillInt(B->matrix());

    multiply(A, B, C, m, n, k, P, r, pattern.cbegin(), divPattern.cbegin());

    output_matrix(A);
    output_matrix(B);
    output_matrix(C);
/*
#ifdef DEBUG
    for( int i = 0; i < P; i++ ) {
        if( getRank() == i ) {

            printf("(%d) A: ", i );
            for( auto j = 0; j < sizeA; j++ )
                printf("%5.3f ", A->matrix()[j] );
            printf("\n");

            printf("(%d) B: ", i );
            for( auto j = 0; j < sizeB; j++ )
                printf("%5.3f ", B->matrix()[j] );
            printf("\n");

            printf("(%d) C: ", i );
            for( auto j = 0; j < sizeC; j++ )
                printf("%5.3f ", C->matrix()[j] );
            printf("\n");
        }
        MPI_Barrier( MPI_COMM_WORLD );
    }
#endif //DEBUG
*/
    MPI_Finalize();

    free(A);
    free(B);
    free(C);

    return 0;
}
