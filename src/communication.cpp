// STL
#include <iostream>

// OpenMP
#include <omp.h>

// Local
#include "communication.h"

// BLAS
#include "blas.h"

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
    omp_set_num_threads(NUM_THREADS);
}

void initCommunication( int *argc, char ***argv ) {
    MPI_Init(argc, argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &commsize );
    omp_set_num_threads(NUM_THREADS);
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
            for (int el = 0; el < size_before[target][bucket]; el++) {
                out[index] = receiving_buffer[dspls[rank] + bucket_offset[rank] + el];
                index++;
            }
            bucket_offset[rank] += size_before[target][bucket];
        }
    }
    std::cout << std::endl;

    std::cout << "Local matrix = " << std::endl;
    for (int i = 0; i < local_size; i++) {
        std::cout << mat[i] << " ";
    }
    std::cout << std::endl;

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

/*
void reduce_old(int div, Interval& P, Interval& newP, Interval& m, Interval& n, double* LC, double* C) {
  // number of processor in each subset
  int subset_size = newP.length();
  // absolute index of the first node of the current group, the group boss
  int Proc0 = P.first();
  // relative index of the current subset, inside the group
  int gp = (rank-Proc0)/subset_size;
  // relative index of the current node, within a subset
  int offset = (rank-Proc0)%subset_size;


  // part of m x n matrix that current rank owns
  Interval localm = m;
  Interval localn = n.subinterval(newP.length(), gp);
  int receive_size = Interval2D(localm, localn.subinterval(div, gp)).size();
  double* receiving_buffer = (double*) malloc(receive_size * (div) *sizeof(double));
  // we send gp-th package to gp-th group, therefore sending size
  // depends on the receiver

  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  std::vector<int> ranks(div);
  std::vector<int> sendcounts(div);
  std::vector<int> sdispls(div);
  std::vector<int> recvcounts(div);
  std::vector<int> rdispls(div);

  int ssum = 0;
  int rsum = 0;
  for (unsigned i = 0; i < div; ++i) {
    int target = Proc0+offset+subset_size*i;
    ranks[i] = target;
    int send_size = Interval2D(localm, localn.subinterval(div, i)).size();
    sendcounts[i] = send_size;
    recvcounts[i] = receive_size;
    sdispls[i] = ssum;
    rdispls[i] = rsum;
    ssum += send_size;
    rsum += receive_size;
  }

  MPI_Group subgroup;
  MPI_Group_incl(world_group, div, ranks.data(), &subgroup);

  MPI_Comm subcomm;
  MPI_Comm_create_group(MPI_COMM_WORLD, subgroup, 0, &subcomm);

  MPI_Alltoallv(LC, sendcounts.data(), sdispls.data(), MPI_DOUBLE, receiving_buffer, recvcounts.data(), rdispls.data(), MPI_DOUBLE, subcomm);

  MPI_Group_free(&world_group);
  MPI_Group_free(&subgroup);
  MPI_Comm_free(&subcomm);

  for (int j = 0; j < receive_size; ++j) {
     int sum = 0;
     for (int i = 0; i < div; ++i) {
       sum += receiving_buffer[i * receive_size + j];
     }
     C[j] = sum;
  }

  free(receiving_buffer);
}
*/

void reduce(int div, Interval& P, Interval& newP, Interval2D& c_range, double* LC, double* C, 
        std::vector<std::vector<int>>& c_current,
        std::vector<int>& c_total_current,
        std::vector<std::vector<int>>& c_expanded,
        std::vector<int>& c_total_expanded,
        MPI_Comm comm) {

    int offset = getRank() - newP.first();
    int gp = (getRank() - P.first()) / newP.length();

    MPI_Comm subcomm;
    MPI_Comm_split(comm, offset, gp, &subcomm);

    int total_size = 0;

    MPI_Request req[div];

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

    int index = 0;
    // go through the communication ring
    for (int i = 0; i < div; i++) {
        int target = i * newP.length() + offset;

        for (int bucket = 0; bucket < n_buckets; ++bucket) {
            for (int el = 0; el < c_current[target][bucket]; ++el) {
                send_buffer[index] = LC[bucket_offset[bucket] + el];
                index++;
            }
            bucket_offset[bucket] += c_current[target][bucket];
        }
    }

    for (int i = 0; i < div; i++) {
        int target = i * newP.length() + offset;
        int send_size = c_total_current[target];
        std::cout << "Rank " << getRank() << " sends " << send_size << " to rank " << i * newP.length() + offset << std::endl;
        MPI_Ireduce(send_buffer.data() + total_size, C, send_size, MPI_DOUBLE, MPI_SUM, i, subcomm, req + i);
        total_size += send_size;
    }

    MPI_Waitall(div, req, MPI_STATUSES_IGNORE);

    MPI_Comm_free(&subcomm);
}
