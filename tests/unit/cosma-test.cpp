// STL
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//Blas
#include <blas.h>

// Local
#include <multiply.hpp>
#include <strategy.hpp>

template<typename T>
void fillInt(T& in) {
  std::generate(in.begin(), in.end(),
    [](){ return (int) (10*drand48()); });
}

bool run(Strategy& s, MPI_Comm comm=MPI_COMM_WORLD, bool one_sided_communication = false) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int m = s.m;
    int n = s.n;
    int k = s.k;
    int P = s.P;

    //Declare A,B and C COSMA matrices objects
    CosmaMatrix A('A', s, rank);
    CosmaMatrix B('B', s, rank);
    CosmaMatrix C('C', s, rank);

    // fill the matrices with random data
    srand48(rank);
    fillInt(A.matrix());
    fillInt(B.matrix());

    // initial sizes
    auto sizeA=A.initial_size();
    auto sizeB=B.initial_size();
    auto sizeC=C.initial_size();

    // fill the matrices with random data
    srand48(rank);
    fillInt(A.matrix());
    fillInt(B.matrix());

    bool isOK;

    //Then rank0 ask for other ranks data
    std::vector<double> As,Bs;
    if (rank==0) {
        As=std::vector<double>(m*k);
        std::copy(A.matrix().cbegin(),A.matrix().cend(),As.begin());
        Bs=std::vector<double>(k*n);
        std::copy(B.matrix().cbegin(),B.matrix().cend(),Bs.begin());

        int offsetA = sizeA;
        int offsetB = sizeB;

        for( int i = 1; i < P; i++ ) {
            int receive_size_A = A.initial_size(i);
            int receive_size_B = B.initial_size(i);
            //Rank 0 receive data
            MPI_Recv(As.data()+offsetA, receive_size_A, MPI_DOUBLE, i, 0, comm,
                MPI_STATUSES_IGNORE);
            MPI_Recv(Bs.data()+offsetB, receive_size_B, MPI_DOUBLE, i, 0, comm,
                MPI_STATUSES_IGNORE);

            offsetA += receive_size_A;
            offsetB += receive_size_B;
        }
    }
    //Rank i send data
    if (rank > 0) {
        MPI_Send(A.matrix_pointer(), sizeA, MPI_DOUBLE, 0, 0, comm);
        MPI_Send(B.matrix_pointer(), sizeB, MPI_DOUBLE, 0, 0, comm);
    }

    MPI_Barrier(comm);

    //Then rank 0 must reorder data locally
    std::vector<double> globA;
    std::vector<double> globB;
    std::vector<double> globCcheck;
    if (rank==0) {
        globA.resize(m*k);
        globB.resize(k*n);
        globCcheck.resize(m*n);
        int offsetA = 0;
        int offsetB = 0;

        for (int i=0; i<P; i++) {
            int local_size_A = A.initial_size(i);
            int local_size_B = B.initial_size(i);

            for (int j=0; j<local_size_A; j++) {
                int y,x;
                std::tie(y,x) = A.global_coordinates(j,i);
                if (y>=0 && x>=0) {
                    globA.at(x*m+y)=As.at(offsetA+j);
                }
            }
            for (int j=0; j<local_size_B; j++) {
                int y,x;
                std::tie(y,x) = B.global_coordinates(j,i);
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

#ifdef DEBUG
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
#endif
    }

    multiply(A, B, C, s, comm, 0.0, one_sided_communication);

    //Then rank0 ask for other ranks data
    std::vector<double> Cs;
    if (rank==0) {
        Cs=std::vector<double>(m*n);
        std::copy(C.matrix().cbegin(),C.matrix().cend(),Cs.begin());

        int offsetC = sizeC;

        for( int i = 1; i < P; i++ ) {
            int receive_size_C = C.initial_size(i);
            //Rank 0 receive data
            MPI_Recv(Cs.data()+offsetC, receive_size_C, MPI_DOUBLE, i, 0, comm,
                MPI_STATUSES_IGNORE);
            offsetC += receive_size_C;
        }
    }
    //Rank i send data
    if (rank > 0) {
        MPI_Send(C.matrix_pointer(), sizeC, MPI_DOUBLE, 0, 0, comm);
    }

    MPI_Barrier(comm);

    //Then rank 0 must reorder data locally
    std::vector<double> globC;
    if (rank==0) {
        globC.resize(m*n);
        int offsetC = 0;

        for (int i=0; i<P; i++) {
            int local_size_C = C.initial_size(i);

            for (int j=0; j<local_size_C; j++) {
                int y,x;
                std::tie(y,x) = C.global_coordinates(j,i);
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
                    std::tie(locidx, rank) = C.local_coordinates(x, y);
                    std::cout<<"global(" << x << ", " << y << ") = (loc = " << locidx << ", rank = " << rank << ") = " << globC.at(i)<<" and should be "<< globCcheck.at(i)<<std::endl;
                }
            }
        } else {
            std::cout <<"Result is OK"<<std::endl;
        }
    }
#ifdef DEBUG
    for( int i = 0; i < P; i++ ) {
        if( rank == i ) {
            printf("(%d) A: ", i );
            for( auto j = 0; j < sizeA; j++ )
                printf("%5.3f ", A.matrix()[j] );
            printf("\n");

            printf("(%d) B: ", i );
            for( auto j = 0; j < sizeB; j++ )
                printf("%5.3f ", B.matrix()[j] );
            printf("\n");

            printf("(%d) C: ", i );
            for( auto j = 0; j < sizeC; j++ )
                printf("%5.3f ", C.matrix()[j] );
            printf("\n");
        }
        MPI_Barrier( comm );
    }
#endif //DEBUG
    return rank > 0 || isOK;
}

int main( int argc, char **argv ) {
    MPI_Init(&argc, &argv);

    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Strategy strategy(argc, argv);

    if (rank == 0) { 
        std::cout << "Strategy = " << strategy << std::endl;
    }

    // first run with two-sided communication backend
    bool isOK = run(strategy, MPI_COMM_WORLD, false);
    MPI_Barrier(MPI_COMM_WORLD);
    // then run it with one-sided communication backend
    isOK = isOK && run(strategy, MPI_COMM_WORLD, true);

    MPI_Finalize();

    return rank==0 ? !isOK : 0;
}
