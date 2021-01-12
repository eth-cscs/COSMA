#include <costa/transform.hpp>
#include <grid2grid/grid_layout.hpp>
#include <grid2grid/transform.hpp>
#include <grid2grid/transformer.hpp>

namespace costa {
template <typename T>
grid2grid::grid_layout<T> get_layout(const ::layout_t *layout) {
    return custom_layout<T>(layout->grid,
                            layout->nlocalblocks, 
                            layout->localblocks);
}

template <typename T>
void transform(
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const T alpha, const T beta,
               // transpose flags
               const char transpose_or_conjugate,
               const MPI_Comm comm
              ) {
    // communicator info
    int P;
    MPI_Comm_size(comm, &P);

    // create grid2grid::grid_layout object from the frontend description
    auto in_layout = get_layout<T>(A);
    auto out_layout = get_layout<T>(B);

    // transform A to B
    grid2grid::transform<T>(in_layout, out_layout, 
            transpose_or_conjugate, alpha, beta, comm);
}

template <typename T>
void transform_multiple(
               const int nlayouts,
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const T* alpha, const T* beta,
               // transpose flags
               const char* trans,
               const MPI_Comm comm
              ) {

    // communicator info
    int P;
    MPI_Comm_size(comm, &P);

    // transformer
    grid2grid::transformer<T> transf(comm);

    // schedule all transforms
    for (int i = 0; i < nlayouts; ++i) {
        // create grid2grid::grid_layout object from the frontend description
        auto in_layout = get_layout<T>(&A[i]);
        auto out_layout = get_layout<T>(&B[i]);

        // schedule the transformation
        transf.schedule(in_layout, out_layout, trans[i], alpha[i], beta[i]);
    }

    // perform the full transformation
    transf.transform();
}

// ***********************************
// template instantiation transform
// ***********************************
template
void transform<int>(
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const int alpha, const int beta,
               // transpose flags
               const char transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform<float>(
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const float alpha, const float beta,
               // transpose flags
               const char transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform<double>(
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const double alpha, const double beta,
               // transpose flags
               const char transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform<std::complex<float>>(
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const std::complex<float> alpha, 
               const std::complex<float> beta,
               // transpose flags
               const char transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform<std::complex<double>>(
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const std::complex<double> alpha, 
               const std::complex<double> beta,
               // transpose flags
               const char transpose_or_conjugate,
               const MPI_Comm comm
              );

// ***********************************
// template instantiation transform
// ***********************************
template
void transform_multiple<int>(
               const int nlayouts,
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const int* alpha, const int* beta,
               // transpose flags
               const char* transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform_multiple<float>(
               const int nlayouts,
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const float* alpha, const float* beta,
               // transpose flags
               const char* transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform_multiple<double>(
               const int nlayouts,
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const double* alpha, const double* beta,
               // transpose flags
               const char* transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform_multiple<std::complex<float>>(
               const int nlayouts,
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const std::complex<float>* alpha, 
               const std::complex<float>* beta,
               // transpose flags
               const char* transpose_or_conjugate,
               const MPI_Comm comm
              );

template
void transform_multiple<std::complex<double>>(
               const int nlayouts,
               const layout_t* A,
               const layout_t* B,
               // scaling parameters
               const std::complex<double>* alpha, 
               const std::complex<double>* beta,
               // transpose flags
               const char* transpose_or_conjugate,
               const MPI_Comm comm
              );
}
