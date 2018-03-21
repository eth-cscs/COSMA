#pragma once

// STL
#include <string>
#include <vector>
#include "interval.hpp"
#include <iostream>
#include <cstring>
#include "communication.h"
#include "blas.h"
#include "matrix.hpp"
#include <omp.h>
#include <semiprof.hpp>

void multiply(CarmaMatrix *A, CarmaMatrix *B, CarmaMatrix *C,
    int m, int n, int k, int P, int r,
    std::string::const_iterator patt, std::vector<int>::const_iterator divPatt);

void multiply(double *A, double *B, double *C,
    Interval& m, Interval& n, Interval& k, Interval& P, int r,
    std::string::const_iterator patt, std::vector<int>::const_iterator divPatt, double beta, 
    MPI_Comm comm);

void local_multiply(double *A, double *B, double *C,
    int m, int n, int k, double beta);

void DFS(double *A, double *B, double *C,
    Interval& m, Interval& n, Interval& k, Interval& P, int r, int divm, int divn, int divk,
    std::string::const_iterator patt, std::vector<int>::const_iterator divPatt, double beta,
    MPI_Comm comm);

void BFS(double *A, double *B, double *C,
    Interval& m, Interval& n, Interval& k, Interval& P, int r, int divm, int divn, int divk,
    std::string::const_iterator patt, std::vector<int>::const_iterator divPatt, double beta,
    MPI_Comm comm);
