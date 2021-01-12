#pragma once

#include <costa/grid2grid/communication_data.hpp>
#include <costa/grid2grid/grid2D.hpp>
#include <costa/grid2grid/grid_layout.hpp>
#include <costa/grid2grid/comm_volume.hpp>

#include <mpi.h>

namespace costa {
// redistribute a single matrix layout (without scaling):
// final_layout = initial_layout
template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               MPI_Comm comm);

// redistribute a single layout with scaling:
// final_layout = beta * final_layout + alpha * initial_layout
template <typename T>
void transform(grid_layout<T> &initial_layout,
               grid_layout<T> &final_layout,
               const char trans,
               const T alpha, const T beta,
               MPI_Comm comm);

// redistribute multiple layouts (without scaling):
// for i in [0, initial_layouts.size()) do:
//     final_layouts[i] = initial_layouts[i]
template <typename T>
void transform(std::vector<layout_ref<T>>& initial_layouts,
               std::vector<layout_ref<T>>& final_layouts,
               MPI_Comm comm);

// redistribute multiple layouts with scaling:
// for i in [0, initial_layouts.size()) do:
//     final_layouts[i] = beta * final_layouts[i] + alpha * initial_layouts[i]
template <typename T>
void transform(std::vector<layout_ref<T>>& initial_layouts,
               std::vector<layout_ref<T>>& final_layouts,
               const char* trans,
               const T* alpha, const T* beta,
               MPI_Comm comm);

// compute the communication volume of transformation
// (creates a graph of communication volumes between ranks)
comm_volume communication_volume(assigned_grid2D& g_init,
                                 assigned_grid2D& g_final);

} // namespace costa
