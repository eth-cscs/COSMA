#include <costa/grid2grid/ranks_reordering.hpp>
#include <algorithm>

std::vector<int> costa::optimal_reordering(comm_volume& comm_volume, int n_ranks, bool& reordered) {
    std::vector<bool> visited(n_ranks, false);

    // identity permutation
    std::vector<int> permutation;
    permutation.reserve(n_ranks);
    for (size_t i = 0; i < n_ranks; ++i) {
        permutation.push_back(i);
    }
    reordered = false;

    std::vector<weighted_edge_t> sorted_edges;
    sorted_edges.reserve(comm_volume.volume.size());
    for (const auto& el : comm_volume.volume) {
        auto& e = el.first;
        int w = el.second;
        int src = e.src;
        int dest = e.dest;

        // w += comm_volume.volume[edge_t{dest, src}];
        if (src == dest) {
            w *= 2;
            ++w;
        }
        w -= comm_volume.volume[edge_t{src, src}];
        w -= comm_volume.volume[edge_t{dest, dest}];

        if (w) {
            sorted_edges.push_back(weighted_edge_t(src, dest, w));
        }
    }

    // sort the edges by weights (decreasing order)
    std::sort(sorted_edges.rbegin(), sorted_edges.rend());

    for (const auto& edge : sorted_edges) {
        // edge: src->dest with weight w
        if (visited[edge.src()] || visited[edge.dest()])
            continue;

        if (edge.weight()) {
            // map src -> dest
            // take this edge to perfect matching
            permutation[edge.src()] = edge.dest();
            permutation[edge.dest()] = edge.src();
            if (edge.src() != edge.dest())
                reordered = true;
        }

        // no adjecent edge to these vertices
        // can be taken in the future
        // to preserve the perfect matching
        visited[edge.src()] = true;
        visited[edge.dest()] = true;
    }

    return permutation;
}


