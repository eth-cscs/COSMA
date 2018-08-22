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
#include <cstdlib>
#include <iostream>
#include <sstream>

//Blas
#include "blas.h"

// Local
#include <multiply.hpp>

template<typename T>
void fillInt(T& in) {
  std::generate(in.begin(), in.end(),
    [](){ return (int) (10*drand48()); });
}

int get_n_iter() {
    const char* value = std::getenv("n_iter");
    std::stringstream strValue;
    strValue << value;

    unsigned int intValue;
    strValue >> intValue;

    if (intValue<0 || intValue > 100) {
        std::cout << "Number of iteration must be in the interval [1, 100]" << std::endl;
        std::cout << "Setting it to 1 iteration instead" << std::endl;
        return 1;
    }

    return intValue;
}

void output_matrix(CosmaMatrix& M, int rank) {
    std::string local = M.which_matrix() + std::to_string(rank) + ".txt";
    std::ofstream local_file(local);
    local_file << M << std::endl;
    local_file.close();
}

long run(Strategy& s, MPI_Comm comm=MPI_COMM_WORLD) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    //Declare A,B and C COSMA matrices objects
    CosmaMatrix A('A', s, rank);
    CosmaMatrix B('B', s, rank);
    CosmaMatrix C('C', s, rank);

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

    if (rank == 0) { 
        std::cout << "Strategy = " << strategy << std::endl;
    }

    MPI_Group group;
    MPI_Comm_group(MPI_COMM_WORLD, &group);
    std::vector<int> exclude_ranks;
    for (int i = strategy.P; i < P; ++i) {
        exclude_ranks.push_back(i);
    }

    MPI_Group new_group;
    MPI_Comm new_comm;

    if (P != strategy.P) {
        MPI_Group_excl(group, exclude_ranks.size(), exclude_ranks.data(), &new_group);
        MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

        if (rank >= strategy.P) {
            MPI_Finalize();
            return 0;
        }
    }

    int n_iter = get_n_iter();
    std::vector<long> times;
    for (int i = 0; i < n_iter+1; ++i) {
        long t_run = 0;
        if (P != strategy.P) 
            t_run = run(strategy, new_comm);
        else 
            t_run = run(strategy);
        if (i == 0) continue;
        times.push_back(t_run);
    }
    std::sort(times.begin(), times.end());

    if (rank == 0) {
        std::cout << "COSMA TIMES [ms] = ";
        for (auto& time : times) {
            std::cout << time << " ";
        }
        std::cout << std::endl;
    }

    if (P != strategy.P) {
        MPI_Group_free(&new_group);
        MPI_Comm_free(&new_comm);
    }

    MPI_Finalize();

    return 0;
}
