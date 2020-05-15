#pragma once

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A local block of the matrix.
 * data: a pointer to the start of the local matrix A_loc
 * ld: leading dimension or distance between two columns of A_loc
 * row: the global block row index
 * col: the global block colum index
 */
struct block {
    void *data;
    const int ld;
    const int row;
    const int col;
};

/**
 * Description of a distributed layout of a matrix
 * rowblocks: number of gobal blocks
 * colblocks: number of gobal blocks
 * rowsplit: [rowsplit[i], rowsplit[i+1]) is range of rows of block i
 * colsplit: [colsplit[i], colsplit[i+1]) is range of columns of block i
 * nlocalblock: number of blocks owned by the current rank
 * localblcoks: an array of block descriptions of the current rank
 */
struct layout {
    int rowblocks;
    int colblocks;
    const int *rowsplit;
    const int *colsplit;
    const int *owners;
    int nlocalblocks;
    block *localblocks;
};

void smultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const float *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const float *beta,
                            const layout *layout_c);

void dmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const double *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const double *beta,
                            const layout *layout_c);

void cmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const float *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const float *beta,
                            const layout *layout_c);

void zmultiply_using_layout(MPI_Comm comm,
                            const char *transa,
                            const char *transb,
                            const double *alpha,
                            const layout *layout_a,
                            const layout *layout_b,
                            const double *beta,
                            const layout *layout_c);

#ifdef __cplusplus
}
#endif
