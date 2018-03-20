// STL
#include <iostream>

// OpenMP
#include <omp.h>

// Local
#include "communication.h"

// BLAS
#include "blas.h"

//#define NUM_THREADS 2

/*
 * We assume that there are P=2^k processors.  At all times a power of
 * 2 processors are working together on a task, and they are contiguously
 * numbered.  That is, if rank i is working on a task with (x-1) other
 * processors, those x processors have
 * ranks floor(i/x)*x,...,(floor(i/x)+1)*x-1
 */

int rank;
int commsize;

int getRank() {
    return rank;
}

int getCommSize() {
    return commsize;
}

void initCommunication() {
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &commsize );
    //omp_set_num_threads(NUM_THREADS);
}

void initCommunication( int *argc, char ***argv ) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    //omp_set_num_threads(NUM_THREADS);
}

int getRelativeRank( int xOld, int xNew ) {
    // the ranks involved in xOld are (rank/xOld)*xOld..(rank/xOld+1)*xOld-1
    return (rank - (rank/xOld)*xOld)/xNew;
}

/**
 * We have N processors, however, we do not need to know the size of
 * the total communication group.
 * Instead, we know that those N processors are already divided into different
 * groups of size x that are ALREADY independant.
 * 
 * We would like to subdivide each of those x-groups into k subsets, and make
 * each of those subset to possess the exact same data as the other subsets.
 * Members of a subset should possess disjoint sets of data. 
 *
 * The size of  each subset is \f$x_h=\frac{x}{k}\f$.
 * Our goal is then to perform a all-to-all global operation,
 * that acts on subsets instead of individual nodes.
 *
 */
void copy_mat( int div, Interval& P, Interval& newP, Interval2D& range, 
        double* mat, double* out, 
        std::vector<std::vector<int>>& size_before,
        std::vector<int>& total_before,
        int total_after, MPI_Comm comm) {

    int offset = getRank() - newP.first();
    int gp = (getRank() - P.first()) / newP.length();

    int local_size = total_before[getRank() - P.first()];

    int sum = 0;
    std::vector<int> total_size(div);
    std::vector<int> dspls(div);
    for (int i = 0; i < div; ++i) {
        int target = i * newP.length() + offset;
        int temp_size = total_before[target];
        dspls[i] = sum;
        sum += temp_size;
        total_size[i] = temp_size;
    }

    std::vector<double> receiving_buffer(total_after);

    MPI_Comm subcomm;
    MPI_Comm_split(comm, offset, gp, &subcomm);

    MPI_Allgatherv(mat, local_size, MPI_DOUBLE, receiving_buffer.data(), 
            total_size.data(), dspls.data(), MPI_DOUBLE, subcomm);

    int n_buckets = size_before[getRank() - P.first()].size();
    int index = 0;
    std::vector<int> bucket_offset(div);
    // order all first DFS parts of all groups first and so on..
    for (int bucket = 0; bucket < n_buckets; bucket++) {
        for (int rank = 0; rank < div; rank++) {
            int target = rank * newP.length() + offset;
            int dsp = dspls[rank] + bucket_offset[rank];
            int b_size = size_before[target][bucket];
            std::copy(receiving_buffer.begin() + dsp, receiving_buffer.begin() + dsp + b_size, out + index);
            index += b_size;
            bucket_offset[rank] += b_size;
        }
    }

#ifdef DEBUG
    std::cout<<"Content of the copied matrix in rank "<<rank<<" is now: "
    <<std::endl;
    for (int j=0; j<sum; j++) {
      std::cout<<out[j]<<" , ";
    }
    std::cout<<std::endl;

#endif
    MPI_Comm_free(&subcomm);
}

void reduce(int div, Interval& P, Interval& newP, Interval2D& c_range, double* LC, double* C,
        std::vector<std::vector<int>>& c_current,
        std::vector<int>& c_total_current,
        std::vector<std::vector<int>>& c_expanded,
        std::vector<int>& c_total_expanded,
        int beta,
        MPI_Comm comm) {

    int offset = getRank() - newP.first();
    int gp = (getRank() - P.first()) / newP.length();

    MPI_Comm subcomm;
    MPI_Comm_split(comm, offset, gp, &subcomm);

    int total_size = 0;

    // reorder the elements as:
    // first all buckets that should be sent to rank 0 then all buckets for rank 1 and so on...
    std::vector<double> send_buffer(c_total_expanded[offset]);
    int n_buckets = c_expanded[offset].size();
    std::vector<int> bucket_offset(n_buckets);

    int sum = 0;
    for (int i = 0; i < n_buckets; ++i) {
        bucket_offset[i] = sum;
        sum += c_expanded[offset][i];
    }

    std::vector<int> recvcnts(div);

    int index = 0;
    // go through the communication ring
    for (int i = 0; i < div; i++) {
        int target = i * newP.length() + offset;
        recvcnts[i] = c_total_current[target];

        for (int bucket = 0; bucket < n_buckets; ++bucket) {
            int b_offset = bucket_offset[bucket];
            int b_size = c_current[target][bucket];
            std::copy(LC + b_offset, LC + b_offset + b_size, send_buffer.begin() + index);
            index += b_size;
            bucket_offset[bucket] += b_size;
        }
    }

    double* receiving_buffer;
    if (beta == 0) {
        receiving_buffer = C;
    } else {
        receiving_buffer = (double*) malloc(sizeof(double) * recvcnts[gp]);
    }

    MPI_Reduce_scatter(send_buffer.data(), receiving_buffer, recvcnts.data(), MPI_DOUBLE, MPI_SUM, subcomm);

    if (beta > 0) {
        for (int i = 0; i < recvcnts[gp]; ++i) {
            C[i] += receiving_buffer[i];
        }
    }

    MPI_Comm_free(&subcomm);
}

