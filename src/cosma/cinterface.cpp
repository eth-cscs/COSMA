#include <costa/grid2grid/grid2D.hpp>

#include <cosma/cinterface.hpp>
#include <cosma/multiply.hpp>

#include <mpi.h>

namespace cosma {

template <class T>
costa::grid_layout<T> grid_from_clayout(int n_ranks,
                                            const ::layout *layout) {

    // Create the local blocks
    std::vector<costa::block<T>> loc_blks;

    // Create blocks
    for (int i = 0; i < layout->nlocalblocks; ++i) {
        auto &block = layout->localblocks[i];
        auto row = block.row;
        auto col = block.col;
        auto ptr = reinterpret_cast<T *>(block.data);
        auto stride = block.ld;

        costa::block_coordinates coord{row, col};
        costa::interval rows{layout->rowsplit[row],
                             layout->rowsplit[row + 1]};
        costa::interval cols{layout->colsplit[col],
                                 layout->colsplit[col + 1]};
        loc_blks.emplace_back(rows, cols, coord, ptr, stride);
    }

    // Grid specification
    std::vector<int> rows_split(layout->rowblocks + 1);
    std::copy_n(layout->rowsplit, rows_split.size(), rows_split.begin());

    std::vector<int> cols_split(layout->colblocks + 1);
    std::copy_n(layout->colsplit, cols_split.size(), cols_split.begin());

    std::vector<std::vector<int>> owners_matrix(layout->rowblocks);
    for (int i = 0; i < layout->rowblocks; ++i) {
        owners_matrix[i].resize(layout->colblocks);
        for (int j = 0; j < layout->colblocks; ++j)
            owners_matrix[i][j] = layout->owners[i * layout->colblocks + j];
    }

    return {{{std::move(rows_split), std::move(cols_split)},
             std::move(owners_matrix),
             n_ranks},
            {std::move(loc_blks)},
            'C'};
}

template <class T>
void xmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const T *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const T *beta,
                            const layout *layout_c) {

    // communicator size and rank
    int rank, P;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    auto cosma_layout_a = grid_from_clayout<T>(P, layout_a);
    auto cosma_layout_b = grid_from_clayout<T>(P, layout_b);
    auto cosma_layout_c = grid_from_clayout<T>(P, layout_c);

    // perform cosma multiplication
    cosma::multiply_using_layout<T>(
        cosma_layout_a, cosma_layout_b, cosma_layout_c, *alpha, *beta, *transa, *transb, comm);
}
} // namespace cosma

#ifdef __cplusplus
extern "C" {
#endif

void smultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const float *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const float *beta,
                            const layout *layout_c) {

    cosma::xmultiply_using_layout<float>(
        comm, transa, transb, alpha, layout_a, layout_b, beta, layout_c);
}

void dmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const double *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const double *beta,
                            const layout *layout_c) {

    cosma::xmultiply_using_layout<double>(
        comm, transa, transb, alpha, layout_a, layout_b, beta, layout_c);
}

void cmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const float *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const float *beta,
                            const layout *layout_c) {

    cosma::xmultiply_using_layout<std::complex<float>>(
        comm,
        transa,
        transb,
        reinterpret_cast<const std::complex<float> *>(alpha),
        layout_a,
        layout_b,
        reinterpret_cast<const std::complex<float> *>(beta),
        layout_c);
}

void zmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const double *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const double *beta,
                            const layout *layout_c) {

    cosma::xmultiply_using_layout<std::complex<double>>(
        comm,
        transa,
        transb,
        reinterpret_cast<const std::complex<double> *>(alpha),
        layout_a,
        layout_b,
        reinterpret_cast<const std::complex<double> *>(beta),
        layout_c);
}

#ifdef __cplusplus
}
#endif
