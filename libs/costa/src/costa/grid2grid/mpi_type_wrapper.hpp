#pragma once

#include <mpi.h>

#include <complex>

namespace costa {

template <typename T>
struct mpi_type_wrapper {};

template <>
struct mpi_type_wrapper<double> {
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

template <>
struct mpi_type_wrapper<float> {
    static MPI_Datatype type() { return MPI_FLOAT; }
};

template <>
struct mpi_type_wrapper<std::complex<double>> {
    static MPI_Datatype type() { return MPI_CXX_DOUBLE_COMPLEX; }
};

template <>
struct mpi_type_wrapper<std::complex<float>> {
    static MPI_Datatype type() { return MPI_CXX_FLOAT_COMPLEX; }
};

template <>
struct mpi_type_wrapper<int> {
    static MPI_Datatype type() { return MPI_INT; }
};

template <>
struct mpi_type_wrapper<int16_t> {
    static MPI_Datatype type() { return MPI_SHORT; }
};

template <>
struct mpi_type_wrapper<char> {
    static MPI_Datatype type() { return MPI_CHAR; }
};

template <>
struct mpi_type_wrapper<unsigned char> {
    static MPI_Datatype type() { return MPI_UNSIGNED_CHAR; }
};

template <>
struct mpi_type_wrapper<unsigned long long> {
    static MPI_Datatype type() { return MPI_UNSIGNED_LONG_LONG; }
};

template <>
struct mpi_type_wrapper<unsigned long> {
    static MPI_Datatype type() { return MPI_UNSIGNED_LONG; }
};

template <>
struct mpi_type_wrapper<bool> {
    static MPI_Datatype type() { return MPI_CXX_BOOL; }
};

template <>
struct mpi_type_wrapper<uint32_t> {
    static MPI_Datatype type() { return MPI_UINT32_T; }
};
} // namespace costa
