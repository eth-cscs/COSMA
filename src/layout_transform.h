// Library
#include "comm.h"
#include "matrixdist.h"
#include "ScalapackMult.h"
#include "util.h"

// Carma
#include "matrix.hpp"

//MPI
#include <mpi.h>

namespace layout_transform {

template<typename T>
T tolerance(int m, int n, T max_value) {
    return 3 * std::max(m, n) * std::numeric_limits<T>::epsilon() * max_value;
}

template<typename T>
bool check(int m, int n, T lhs, T rhs, T tol) {
    return std::abs(lhs - rhs) <= tol;
}

template<typename T, Mapping MScalapack>
void scalapack_to_carma(CarmaMatrix& carma_mat, ScalapackMatrixDist<T, MScalapack>& mat, ScalapackComm2D<MScalapack>& comm) {
    int matSize = mat.local_m() * mat.local_n();
    // flatten the scalapack local matrix to a standard vector
    double* scalapack_local_matrix;
    MPI_Alloc_mem(sizeof(double)*matSize, MPI_INFO_NULL, &scalapack_local_matrix);
    for (int li = 0; li < mat.local_m(); li++) {
        for (int lj = 0; lj < mat.local_n(); lj++) {
            int gi, gj;
            std::tie(gi, gj) = mat.getGlobalIndex(li, lj);
            if (gi < 0 || gj < 0 || gi >= mat.m() || gj >= mat.n()) {
                continue;
            }
            scalapack_local_matrix[li * mat.local_n() + lj] = *mat.ptr(li, lj);
        }
    }
    // mpi window containing the local matrix that should be accessible to all the ranks
    MPI_Win win;

    MPI_Win_create(scalapack_local_matrix, matSize*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    // between create and free, the local matrix window is accessible by all processes in the comm_world

    // acts as a barrier and starts an epoch of data exchange
    MPI_Win_fence(0, win);

    for (int locIdx = 0; locIdx < carma_mat.size(); locIdx++) {
        int gi, gj;
        std::tie(gi, gj) = carma_mat.global_coordinates(locIdx, getRank());
        if (gi < 0 || gj < 0 || gi >= mat.m() || gj >= mat.n()) {
            continue;
        }
        LocalIndex2D scalapack_locIdx = mat.getLocalIndex(gi, gj);
        int li = scalapack_locIdx.indexRow().index();
        int lj = scalapack_locIdx.indexCol().index();
        int rank_col, rank_row;
        std::tie(rank_row, rank_col) = scalapack_locIdx.owner();
        int rank = comm.rankFromCoords(rank_row, rank_col);
      /*
#ifdef DEBUG
        std::cout << "Rank " << rank << " has coordinates (" << rank_row << ", " << rank_col << ")" << std::endl;
        std::cout << "Getting (" << gi << ", " << gj << ") from rank " << rank << " from position " << li * mat.local_n(rank_col) + lj << std::endl;
#endif
       */
        MPI_Get(carma_mat.matrix_pointer() + locIdx, 1, MPI_DOUBLE, rank, li * mat.local_n(rank_col) + lj, 1, MPI_DOUBLE, win);
    }
    // acts as a barrier and closes the epoch of data exchange
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);
    MPI_Free_mem(scalapack_local_matrix);
}

template<typename T, Mapping MScalapack>
void carma_to_scalapack(CarmaMatrix& carma_mat, ScalapackMatrixDist<T, MScalapack>& mat, ScalapackComm2D<MScalapack>& comm) {
    int matSize = mat.local_m() * mat.local_n();
    //std::cout << "Rank " << getRank() << ", matSize = " << matSize << std::endl;
    //std::vector<T> scalapack_local_matrix(matSize);
    double* scalapack_local_matrix;
    MPI_Alloc_mem(sizeof(double)*matSize, MPI_INFO_NULL, &scalapack_local_matrix);

    // mpi window containing the local matrix that should be accessible to all the ranks
    MPI_Win win;

    MPI_Win_create(scalapack_local_matrix, matSize*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    // between create and free, the local matrix window is accessible by all processes in the comm_world

    // acts as a barrier and starts an epoch of data exchange
    MPI_Win_fence(0, win);

    for (int locIdx = 0; locIdx < carma_mat.size(); locIdx++) {
        int gi, gj;
        std::tie(gi, gj) = carma_mat.global_coordinates(locIdx, getRank());
        if (gi < 0 || gj < 0 || gi >= mat.m() || gj >= mat.n()) {
            continue;
        }
        LocalIndex2D scalapack_locIdx = mat.getLocalIndex(gi, gj);
        int li = scalapack_locIdx.indexRow().index();
        int lj = scalapack_locIdx.indexCol().index();
        int rank_col, rank_row;
        std::tie(rank_row, rank_col) = scalapack_locIdx.owner();
        int rank = comm.rankFromCoords(rank_row, rank_col);
      /*
#ifdef DEBUG
        std::cout << "Rank " << rank << " has coordinates (" << rank_row << ", " << rank_col << ")" << std::endl;
        std::cout << "Putting (" << gi << ", " << gj << ") on rank " << rank << " at position " << locIdx << std::endl;
#endif
       */
        MPI_Put(carma_mat.matrix_pointer() + locIdx, 1, MPI_DOUBLE, rank, li * mat.local_n(rank_col) + lj, 1, MPI_DOUBLE, win);
    }
    // acts as a barrier and closes the epoch of data exchange
    MPI_Win_fence(0, win);

    //std::cout << "Exited a fence!\n";
    // unflatten the received data
    for (int li = 0; li < mat.local_m(); li++) {
        for (int lj = 0; lj < mat.local_n(); lj++) {
            int gi, gj;
            std::tie(gi, gj) = mat.getGlobalIndex(li, lj);
            if (gi < 0 || gj < 0 || gi >= mat.m() || gj >= mat.n()) {
                continue;
            }
            *mat.ptr(li, lj) = scalapack_local_matrix[li * mat.local_n() + lj];
        }
    }

    MPI_Win_free(&win);
    MPI_Free_mem(scalapack_local_matrix);
    //std::cout << "Freed all the memory\n";
}

// compares 2 scalapack matrices
template<typename T, Mapping mapping>
bool compare_matrices(ScalapackMatrixDist<T, mapping>& mat1, ScalapackMatrixDist<T, mapping>& mat2, std::string mat) {
    int m = mat1.m();
    int n = mat1.n();
    T max_el = mat1.max_local_element();
    bool equal = true;

    T tol = tolerance(m, n, max_el);

#ifdef DEBUG
    std::cout << "Comparing matrices in Scalapack layout with tolerance: " << tol <<  std::endl;
    std::cout << "Global matrix size: (" << m << ", " << n << ")\n";
    std::cout << "Local matrix sizes: (" << mat1.local_m() << ", " << mat1.local_n() << ") and (" << mat2.local_m() << ", " << mat2.local_n() << ")" << std::endl;
#endif

    if ((mat1.local_m() != mat2.local_m()) || (mat1.local_n() != mat2.local_n()) ) {
        std::cout << "Warning: local matrices are not of the same size. Comparing only the common prefix.\n";
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            LocalIndex2D local1 = mat1.getLocalIndex(i, j);
            LocalIndex2D local2 = mat2.getLocalIndex(i, j);
            if (local1.isMine() && local2.isMine()) {
                bool ok = check(m, n, mat1(i, j), mat2(i, j), tol);
                equal &= ok;
                if (!ok) {
                    std::cout << "Rank " << getRank() << ": Scalapack: " << mat << "[" + std::to_string(i) + ", " + std::to_string(j) + "] = " + std::to_string(mat1(i, j)) + " should be equal to " + std::to_string(mat2(i, j)) + "\n";
                }
            }
        }
    }

    if (equal) {
        std::cout << "Matrices " + mat << " are equal up to the absolute tolerance " << tol << std::endl;
    }
    else {
        std::cout << "Matrices " << mat << " are NOT equal!\n";
    }
    return equal;
}

// compares 2 carma matrices
template<typename T>
bool compare_matrices(CarmaMatrix& mat1, CarmaMatrix& mat2, std::string mat) {
    int length = std::min(mat1.initial_size(), mat2.initial_size());
    int m = mat1.m();
    int n = mat1.n();
    T max_el = *max_element(mat1.matrix().begin(), mat1.matrix().end());
    T tol = tolerance(m, n, max_el);
    bool equal = true;

#ifdef DEBUG
    std::cout << "Comparing matrices in CARMA layout with tolerance: " << tol << std::endl;
    std::cout << "Global matrix size: (" << m << ", " << n << ")\n";
    std::cout << "Local matrix sizes: " << mat1.initial_size() << " and " << mat2.initial_size() << std::endl;
#endif

    if (mat1.initial_size() != mat2.initial_size()) {
        std::cout << "Warning: local matrices are not of the same size. Comparing only the common prefix.\n";
    }

    for (int i = 0; i < length; i++) {
        int gi, gj;
        std::tie(gi, gj) = mat1.global_coordinates(i, getRank());
        if( gi < 0 || gj < 0 || gi >= m || gj >= n) {
            continue;
        }

        bool ok = check(m, n, mat1.matrix()[i], mat2.matrix()[i], tol);
        equal &= ok;

        if (!ok) {
            std::cout << "CARMA: " << mat << "[" + std::to_string(gi) + ", " + std::to_string(gj) + "] = " + std::to_string(mat1.matrix()[i]) + " should be equal to " + std::to_string(mat2.matrix()[i]) + "\n";
        }
    }

    if (equal) {
        std::cout << "Matrices " + mat << " are equal up to the absolute tolerance " << tol << std::endl;
    }
    else {
        std::cout << "Matrices " << mat << " are NOT equal!\n";
    }

    return equal;
}
} // namespace layout_transform

