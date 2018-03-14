// STL
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

template<typename T>
void fillInt(T& in) {
  std::generate(in.begin(), in.end(),
    [](){ return (int) (10*drand48()); });
}

int main( int argc, char **argv ) {
    initCommunication(&argc, &argv);

    auto m = read_int(argc, argv, "-m", 4096);
    auto n = read_int(argc, argv, "-n", 4096);
    auto k = read_int(argc, argv, "-k", 4096);
    auto r = read_int(argc, argv, "-r", 4);
    char* patt = read_string(argc, argv, "-p", "bbbb");
    std::string pattern(patt);
    char* divPatternStr = read_string(argc, argv, "-d", "211211211211");
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

    std::cout << "P = " << P << std::endl;

    // Check if the parameters make sense.
    int m0 = m;
    int n0 = n;
    int k0 = k;
    int prodBFS = 1;

    for (int i = 0; i < r; i++) {
        int divM = divPattern[3*i];
        int divN = divPattern[3*i + 1];
        int divK = divPattern[3*i + 2];

        m0 /= divM;
        n0 /= divN;
        k0 /= divK;

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

    //Then rank0 ask for other ranks data
    std::vector<double> As,Bs;
    if (getRank()==0) {
        As=std::vector<double>(m*k);
        std::copy(A->matrix().cbegin(),A->matrix().cend(),As.begin());
        Bs=std::vector<double>(k*n);
        std::copy(B->matrix().cbegin(),B->matrix().cend(),Bs.begin());

        int offsetA = sizeA;
        int offsetB = sizeB;

        for( int i = 1; i < P; i++ ) {
            int receive_size_A = A->initial_size(i); 
            int receive_size_B = B->initial_size(i);
            //Rank 0 receive data
            MPI_Recv(As.data()+offsetA, receive_size_A, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                nullptr);
            MPI_Recv(Bs.data()+offsetB, receive_size_B, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                nullptr);

            offsetA += receive_size_A;
            offsetB += receive_size_B;
        }
    }
    //Rank i send data
    if (getRank() > 0) {
        MPI_Send(A->matrix_pointer(), sizeA, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(B->matrix_pointer(), sizeB, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier( MPI_COMM_WORLD );

    //Then rank 0 must reorder data locally
    std::vector<double> globA;
    std::vector<double> globB;
    std::vector<double> globCcheck;
    if (getRank()==0) {
        globA.resize(m*k);
        globB.resize(k*n);
        globCcheck.resize(m*n);
        int offsetA = 0;
        int offsetB = 0;

        for (int i=0; i<P; i++) {
            int local_size_A = A->initial_size(i);
            int local_size_B = B->initial_size(i);
            int local_size_C = C->initial_size(i);

            for (int j=0; j<local_size_A; j++) {
                int y,x;
                std::tie(y,x) = A->global_coordinates(j,i);
                if (y>=0 && x>=0) {
                    globA.at(x*m+y)=As.at(offsetA+j);
                }
            }
            for (int j=0; j<local_size_B; j++) {
                int y,x;
                std::tie(y,x) = B->global_coordinates(j,i);
                //std::cout << "Mapped successfully!\n";
                if (y>=0 && x>=0) {
                    //globB.at(x*n+y)=Bs.at(i*sizeB+j);
                    //std::cout << "Retrieved Bs value successfully!\n";
                    globB.at(x*k+y) = Bs.at(offsetB+j);
                }
            }

            offsetA += local_size_A;
            offsetB += local_size_B;
        }
        //Now compute the result
        char N = 'N';
        double one = 1., zero = 0.;
        dgemm_(&N, &N, &m, &n, &k, &one, globA.data(), &m, globB.data(), &k, &zero,
            globCcheck.data(), &m);

        std::cout << "Complete matrix A: " << std::endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                std::cout << globA[j * m + i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "Complete matrix B: " << std::endl;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << globB[j * k + i] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Complete matrix C: " << std::endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << globCcheck[j * m + i] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    multiply(A, B, C, m, n, k, P, r, pattern.cbegin(), divPattern.cbegin());
    MPI_Barrier(MPI_COMM_WORLD);

    //Then rank0 ask for other ranks data
    std::vector<double> Cs;
    if (getRank()==0) {
        Cs=std::vector<double>(m*n);
        std::copy(C->matrix().cbegin(),C->matrix().cend(),Cs.begin());

        int offsetC = sizeC;

        for( int i = 1; i < P; i++ ) {
            int receive_size_C = C->initial_size(i); 
            //Rank 0 receive data
            MPI_Recv(Cs.data()+offsetC, receive_size_C, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                nullptr);
            offsetC += receive_size_C;
        }
    }
    //Rank i send data
    if (getRank() > 0) {
        MPI_Send(C->matrix_pointer(), sizeC, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier( MPI_COMM_WORLD );

    //Then rank 0 must reorder data locally
    std::vector<double> globC;
    if (getRank()==0) {
        globC.resize(m*n);
        int offsetC = 0;

        for (int i=0; i<P; i++) {
            int local_size_C = C->initial_size(i);

            for (int j=0; j<local_size_C; j++) {
                int y,x;
                std::tie(y,x) = C->global_coordinates(j,i);
                if (y>=0 && x>=0) {
                    globC.at(x*m+y)=Cs.at(offsetC+j);
                }
            }
            offsetC += local_size_C;
        }

        auto max_globCcheck = *(max_element(globCcheck.begin(), globCcheck.end()));
        //Now Check result
        isOK = std::inner_product(globCcheck.cbegin(), globCcheck.cend(),
            globC.cbegin(), true,
            [](bool lhs, bool rhs){ return lhs && rhs; },
            [=](double lhs, double rhs){ return std::abs(lhs - rhs) / max_globCcheck <= 3 * k * std::numeric_limits<double>::epsilon();});
        if (!isOK) {
            std::cout <<"Result is NOT OK"<<std::endl;
            for (int i=0; i<m*n; i++) {
                if (globCcheck[i] != globC[i]) {
                    int x = i % m;
                    int y = i / m;
                    int locidx, rank;
                    std::tie(locidx, rank) = C->local_coordinates(x, y);
                    std::cout<<"global(" << x << ", " << y << ") = (loc = " << locidx << ", rank = " << rank << ") = " << globC.at(i)<<" and should be "<< globCcheck.at(i)<<std::endl;
                }
            }
        } else {
            std::cout <<"Result is OK"<<std::endl;
        }
    }

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


    MPI_Finalize();

    free(A);
    free(B);
    free(C);

    if (getRank() == 0) {
        return !isOK;
    }

    return 0;
}
