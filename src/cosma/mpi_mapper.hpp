#pragma once

#include <complex>
#include <mpi.h>

namespace cosma {

/**
 * Maps a primitive numeric type to a MPI type.
 *
 * @tparam Scalar the numeric type to be mapped
 */
template <typename Scalar>
struct mpi_mapper {
  static inline MPI_Datatype getType();
};

template <>
inline MPI_Datatype mpi_mapper<double>::getType() {
  return MPI_DOUBLE;
}

template <>
inline MPI_Datatype mpi_mapper<float>::getType() {
  return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_mapper<std::complex<float>>::getType() {
  return MPI_CXX_FLOAT_COMPLEX;
}

template <>
inline MPI_Datatype mpi_mapper<std::complex<double>>::getType() {
  return MPI_CXX_DOUBLE_COMPLEX;
}

// Removes const qualifier
//
template <typename Scalar>
struct mpi_mapper<const Scalar> {
  static inline MPI_Datatype getType();
};

template <typename Scalar>
inline MPI_Datatype mpi_mapper<const Scalar>::getType() {
  return mpi_mapper<Scalar>::getType();
}

} // end namespace cosma
