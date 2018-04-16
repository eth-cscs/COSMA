#include "communicator.hpp"
#include "strategy.hpp"
#include <tuple>
#include <mpi.h>

// every communication pair represents one edge in the topology graph
// this functions finds all edges for the current rank
// weight of the edge is given by the amount of communicated data
void get_edges(int rank, const Strategy& strategy, std::vector<int>& dest, std::vector<int>& weight);

// creates the graph that represents the topology of mpi communicator
// it is "aware" of all the communications that will happen throughout 
MPI_Comm adapted_communicator(MPI_Comm comm, const Strategy& strategy);

