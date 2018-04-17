#ifndef _COMMUNICATOR_H_
#define _COMMUNICATOR_H_

#include <mpi.h>
#include <stdlib.h>
#include "interval.hpp"
#include <algorithm>
#include <iostream>
#include "blas.h"
#include <tuple>

namespace communicator {
    void initialize(int * argc, char ***argv);

    void free(MPI_Comm comm = MPI_COMM_WORLD);
    void free_group(MPI_Group comm_group);

    int rank(MPI_Comm comm = MPI_COMM_WORLD);

    int size(MPI_Comm comm = MPI_COMM_WORLD);

    void finalize();

    void barrier(MPI_Comm comm = MPI_COMM_WORLD);

    int offset(Interval& P, int div, int r = rank());

    int group(Interval& P, int div, int r = rank());

    std::pair<int, int> group_and_offset(Interval& P, int div, int r = rank());

    /*
      We split P processors into div groups of P/div processors.
        * gp from [0..(div-1)] is the id of the group of the current rank
        * offset from [0..(newP.length()-1)] is the offset of current rank inside its group

      We then define the communication ring of the current processor as:
        i * (P/div) + offset, where i = 0..(div-1) and offset = rank() - i * (P/div)
    */
    MPI_Comm split_in_groups(MPI_Comm comm, Interval& P, int div, int r = rank());

    MPI_Comm split_in_comm_rings(MPI_Comm comm, Interval& P, int div, int r = rank());

    MPI_Comm create_comm_ring(MPI_Comm comm, MPI_Group comm_group,
                              std::vector<int>& ranks,
                              Interval& P, int div, int r = rank());

    int rank_inside_ring(Interval& P, int div, int global_rank=rank());

    int rank_outside_ring(Interval& P, int div, int off, int i);

    int relative_rank(Interval& P, int r = rank());

    void copy(int div, Interval& P, double* in, double* out,
              std::vector<std::vector<int>>& size_before,
              std::vector<int>& total_before,
              int total_after, MPI_Comm comm, MPI_Group comm_group);

    void reduce(int div, Interval& P, double* LC, double* C,
                std::vector<std::vector<int>>& c_current,
                std::vector<int>& c_total_current,
                std::vector<std::vector<int>>& c_expanded,
                std::vector<int>& c_total_expanded,
                int beta,
                MPI_Comm comm, MPI_Group comm_group);
};
#endif
