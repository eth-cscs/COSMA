#include <cosma/multiply.hpp>
#include <cosma/context.hpp>
#include <cosma/multiply_using_layout.hpp>
#include <cosma/strategy.hpp>
#include <cosma/matrix.hpp>

namespace cosma {

using zfloat_t = std::complex<float>;
using zdouble_t = std::complex<double>;

template <typename T>
void multiply_using_layout(
              grid2grid::grid_layout<T> A,
              grid2grid::grid_layout<T> B,
              grid2grid::grid_layout<T> C,
              int m, int n, int k,
              T alpha, T beta,
              MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // find an optimal strategy for this problem
    Strategy strategy(m, n, k, P);

    // create COSMA matrices
    CosmaMatrix<T> A_cosma('A', strategy, rank);
    CosmaMatrix<T> B_cosma('B', strategy, rank);
    CosmaMatrix<T> C_cosma('C', strategy, rank);

    // get abstract layout descriptions for COSMA layout
    auto cosma_layout_a = A_cosma.get_grid_layout();
    auto cosma_layout_b = B_cosma.get_grid_layout();
    auto cosma_layout_c = C_cosma.get_grid_layout();

    // transform A and B from given layout to cosma layout
    grid2grid::transform<T>(A, cosma_layout_a, comm);
    grid2grid::transform<T>(B, cosma_layout_b, comm);

    // transform C from given layout to cosma layout only if beta > 0
    if (std::abs(beta) > 0) {
        grid2grid::transform<T>(C, cosma_layout_c, comm);
    }

    // perform cosma multiplication
    auto ctx = cosma::make_context();
    multiply<T>(ctx, A_cosma, B_cosma, C_cosma, strategy, comm, alpha, beta);

    // transform the result from cosma back to the given layout
    grid2grid::transform<T>(cosma_layout_c, C, comm);
}

// explicit instantiation
template void multiply_using_layout<double>(
                  grid2grid::grid_layout<double> A,
                  grid2grid::grid_layout<double> B,
                  grid2grid::grid_layout<double> C,
                  int m, int n, int k,
                  double alpha, double beta,
                  MPI_Comm comm);

template void multiply_using_layout<float>(
                  grid2grid::grid_layout<float> A,
                  grid2grid::grid_layout<float> B,
                  grid2grid::grid_layout<float> C,
                  int m, int n, int k,
                  float alpha, float beta,
                  MPI_Comm comm);

template void multiply_using_layout<zdouble_t>(
                  grid2grid::grid_layout<zdouble_t> A,
                  grid2grid::grid_layout<zdouble_t> B,
                  grid2grid::grid_layout<zdouble_t> C,
                  int m, int n, int k,
                  zdouble_t alpha, zdouble_t beta,
                  MPI_Comm comm);

template void multiply_using_layout<zfloat_t>(
                  grid2grid::grid_layout<zfloat_t> A,
                  grid2grid::grid_layout<zfloat_t> B,
                  grid2grid::grid_layout<zfloat_t> C,
                  int m, int n, int k,
                  zfloat_t alpha, zfloat_t beta,
                  MPI_Comm comm);

}
