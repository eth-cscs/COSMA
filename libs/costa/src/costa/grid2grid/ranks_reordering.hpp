#pragma once

#include <costa/grid2grid/comm_volume.hpp>
#include <vector>

namespace costa {
std::vector<int> optimal_reordering(comm_volume& comm_volume, int n_ranks, bool& reordered);
}


