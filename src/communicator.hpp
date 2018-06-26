#ifndef _COMMUNICATOR_H_
#define _COMMUNICATOR_H_
#include "blas.h"
#include "interval.hpp"
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <tuple>
#include "strategy.hpp"

class communicator {
public:
    communicator(const Strategy& strategy, MPI_Comm comm=MPI_COMM_WORLD);
    ~communicator();

    virtual void copy(Interval& P, double* in, double* out,
        std::vector<std::vector<int>>& size_before,
        std::vector<int>& total_before,
        int total_after, int step) = 0;

    virtual void reduce(Interval& P, double* in, double* out,
        std::vector<std::vector<int>>& c_current,
        std::vector<int>& c_total_current,
        std::vector<std::vector<int>>& c_expanded,
        std::vector<int>& c_total_expanded,
        int beta, int step) = 0;

    virtual void synchronize() = 0;

    // adds two vectors of size n and stores the result in a (a += b)
    void add(double* a, double* b, int n);

    // creates the graph that represents the topology of mpi communicator
    // it is "aware" of all the communications that will happen throughout
    void add_topology();

    // invokes MPI_Init
    static void initialize(int * argc, char ***argv);

    // rank in the initial communicator
    int rank();
    int relative_rank(Interval& P);
    int offset(Interval& P, int div);
    int group(Interval& P, int div);
    std::pair<int, int> group_and_offset(Interval& P, int div);
    int rank_inside_ring(Interval& P, int div);

    // barrier over all the ranks taking part in the multiplication
    void full_barrier();
    // barrier over the active communicator in step 
    void barrier(int step);

    // communicator active in step
    MPI_Comm active_comm(int step);

    // size of the initial communicator
    int comm_size();

    static void free_comm(MPI_Comm& comm);
    static void free_group(MPI_Group& comm_group);

    static void finalize();

protected:
    std::vector<MPI_Comm> comm_ring_;
    std::vector<MPI_Comm> comm_subproblem_;
    int rank_;
    const Strategy& strategy_;
    std::vector<int> step_to_comm_index_;
    MPI_Comm full_comm_;
    int comm_size_;

    static int relative_rank(Interval& P, int rank);
    static int offset(Interval& P, int div, int rank);
    static int group(Interval& P, int div, int rank);
    static std::pair<int, int> group_and_offset(Interval& P, int div, int rank);

    void get_topology_edges(std::vector<int>& dest, std::vector<int>& weight);

    /*
       We split P processors into div groups of P/div processors.
     * gp from [0..(div-1)] is the id of the group of the current rank
     * offset from [0..(newP.length()-1)] is the offset of current rank inside its group

     We then define the communication ring of the current processor as:
     i * (P/div) + offset, where i = 0..(div-1) and offset = rank() - i * (P/div)
     */
    static int rank_inside_ring(Interval& P, int div, int global_rank);
    static int rank_outside_ring(Interval& P, int div, int off, int i);

    void create_communicators(MPI_Comm comm);
    // same as create just uses MPI_Comm_split instead of MPI_Comm_create
    void split_communicators(MPI_Comm comm);
    MPI_Comm create_comm_ring(MPI_Comm comm, Interval& P, int offset, int div);
    MPI_Comm create_comm_subproblem(MPI_Comm comm, Interval& P, Interval& newP);

    void free_comms();
};
#endif
