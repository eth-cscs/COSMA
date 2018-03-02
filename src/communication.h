#include <mpi.h>
#include <stdlib.h>
#include "interval.hpp"
#include "library.h"

#ifndef COMMUNICATION_H
#define COMMUNICATION_H

int getRank();
int getCommSize();
void initCommunication( int *argc, char ***argv );
void initCommunication();
int getRelativeRank( int xOld, int xNew );

void copy_mat(int div, Interval& P, Interval& newP, Interval2D& range, double* mat, double* out, std::vector<std::vector<int>>& bucket_sizes, std::vector<int>& total_sizes, int total_after, MPI_Comm comm);

void reduce(int div, Interval& P, Interval& newP, Interval2D& c_range, double* LC, double* C, 
        std::vector<std::vector<int>>& c_current,
        std::vector<int>& c_total_current,
        std::vector<std::vector<int>>& c_expanded,
        std::vector<int>& c_total_expanded,
        MPI_Comm comm);

#endif 
