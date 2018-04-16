#include "topology.hpp"

void get_edges(int rank, const Strategy& strategy,
        std::vector<int>& dest, std::vector<int>& weight) {
    int m = strategy.m;
    int n = strategy.n;
    int k = strategy.k;
    Interval P(0, strategy.P-1);
    int n_steps = strategy.n_steps;

    for (int step = 0; step < n_steps; ++step) {
        m /= strategy.divisor_m(step);
        n /= strategy.divisor_n(step);
        k /= strategy.divisor_k(step);

        if (strategy.bfs_step(step)) {
            int div = strategy.divisor(step);
            int partition_idx = P.partition_index(div, rank);
            Interval newP = P.subinterval(div, partition_idx);
            int group, offset;
            std::tie(group, offset) = communicator::group_and_offset(P, div, rank);

            for (int gp = 0; gp < div; ++gp) {
                int neighbor = P.first() + communicator::rank_outside_ring(P, div, offset, gp);
                if (neighbor == rank) continue;
                dest.push_back(neighbor);

                int communication_size = 0;
                if (strategy.split_A(step))
                    communication_size = m * k;
                else if (strategy.split_B(step))
                    communication_size = k * n;
                else
                    communication_size = m * n;

                weight.push_back(communication_size);
            }

            P = newP;
        }
    }
}

MPI_Comm adapted_communicator(MPI_Comm comm, const Strategy& strategy) {
    MPI_Comm graph;
    int rank = communicator::rank();
    int n_sources = 1;
    int source[1] = {rank};
    std::vector<int> dest;
    std::vector<int> weight;

    get_edges(rank, strategy, dest, weight);

    int n_edges = dest.size();
    int degree = dest.size();
    int degrees[1] = {degree};

    if (n_edges < 1)
        return comm;

    MPI_Dist_graph_create(comm, n_sources, source,
            degrees, dest.data(), weight.data(), MPI_INFO_NULL, true, &graph);
    /*
    std::vector<int> temp_dest;
    std::vector<int> all_dest;
    std::vector<int> all_degree;

    std::vector<int> weight;

    for (int i = 0; i < communicator::size(comm); ++i) {
        temp_dest = std::vector<int>();
        weight = std::vector<int>();
        get_edges(i, strategy, temp_dest, weight);
        all_dest.resize(all_dest.size() + temp_dest.size());
        for (auto& v : temp_dest) {
            all_dest.push_back(v);
        }
        all_degree.push_back(temp_dest.size() + ((i > 0) ? all_degree[i-1] : 0));
    }

    MPI_Graph_create(comm, communicator::size(comm), all_degree.data(), all_dest.data(), true, &graph);

    int newrank;
    MPI_Graph_map(comm, communicator::size(comm), all_degree.data(), all_dest.data(), &newrank);

    std::cout << "rank " << communicator::rank() << " is now " << newrank << std::endl;
    */
    return graph;
}


