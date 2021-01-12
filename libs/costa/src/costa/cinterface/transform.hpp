#pragma once
#include <costa/layout.hpp>
#include <complex>
#include <mpi.h>

/**
 * The general transformation looks as follows (A = input, B = output):
 * B = alpha * (A^T) + beta * B
 * The parameters are as follows:
 * alpha: a scalar to be multiplied by the input matrix
 * beta: a scalar to be multiplied by the output matrix
 * transpose_or_conjugate ('N', 'T' or 'C'): describes whether the input
 * matrix should be left unchanged ('N'), transposed ('T') or conjugated ('C')
 */
namespace costa {
// applies the transformation: B = alpha * (A^T) + beta * B
template <typename T>
void transform(
               // A = input layout, B = output layout
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const T alpha, const T beta,
               // transpose flags
               const char transpose_or_conjugate,
               // communicator
               const MPI_Comm comm
              );

// transforming multiple layouts at once (in a single communication round), 
// minimizing the latency. In addition, the relabelling will take into account
// the transformations of all layouts at once.
// Due to the minimized communication latency and also 
// due to the optimal relabelling accross all transformations, 
// this is potentially more efficient than invoking
// single transform for each layout separately.
template <typename T>
void transform_multiple(
               // number of layouts to transform
               const int nlayouts,
               // array of input layouts of size nlayouts
               const layout_t* A,
               // array of output layouts of size nlayouts
               const layout_t* B,
               // scaling parameter array of size nlayouts
               const T* alpha, const T* beta,
               // transpose flags array of size nlayouts
               const char* transpose_or_conjugate,
               // communicator
               const MPI_Comm comm
              );
}
