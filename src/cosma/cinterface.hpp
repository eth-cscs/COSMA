#pragma once

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

struct layout {
    int rowblocks;
    int colblocks;
    const int *rowsplit;
    const int *colsplit;
    const int *owners;
    int nlocalblock;
    void **localblock_data;
    const int *localblock_ld;
    const int *localblock_row;
    const int *localblock_col;
};

void psmultiply(MPI_Comm comm,
                const char *transa,
                const char *transb,
                const float *alpha,
                const layout *layout_a,
                const layout *layout_b,
                const float *beta,
                const layout *layout_c);

void pdmultiply(MPI_Comm comm,
                const char *transa,
                const char *transb,
                const double *alpha,
                const layout *layout_a,
                const layout *layout_b,
                const double *beta,
                const layout *layout_c);

void pcmultiply(MPI_Comm comm,
                const char *transa,
                const char *transb,
                const float *alpha,
                const layout *layout_a,
                const layout *layout_b,
                const float *beta,
                const layout *layout_c);

void pzmultiply(MPI_Comm comm,
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
