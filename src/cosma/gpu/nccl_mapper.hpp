#pragma once

#include <complex>

#if defined(TILED_MM_CUDA)
#include <nccl.h>

#elif defined(TILED_MM_ROCM)
#include <rccl/rccl.h>

#else
#error Either TILED_MM_CUDA or TILED_MM_ROCM must be defined!
#endif

namespace cosma {
namespace gpu {
/**
 * Maps a primitive numeric type to a MPI type.
 *
 * @tparam Scalar the numeric type to be mapped
 */
template <typename Scalar>
struct nccl_mapper {
  static inline ncclDataType_t getType();
};

template <>
inline ncclDataType_t nccl_mapper<double>::getType() {
  return ncclDouble;
}

template <>
inline ncclDataType_t nccl_mapper<float>::getType() {
  return ncclFloat;
}

template <>
inline ncclDataType_t nccl_mapper<std::complex<double>>::getType() {
  return ncclDouble;
}

template <>
inline ncclDataType_t nccl_mapper<std::complex<float>>::getType() {
  return ncclFloat;
}

// Removes const qualifier
//
template <typename Scalar>
struct nccl_mapper<const Scalar> {
  static inline ncclDataType_t getType();
};

template <typename Scalar>
inline ncclDataType_t nccl_mapper<const Scalar>::getType() {
  return nccl_mapper<Scalar>::getType();
}

} // end namespace gpu
} // end namespace cosma
