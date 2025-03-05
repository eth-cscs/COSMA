find_package("nvpl_blas" REQUIRED)
find_package("nvpl_lapack" REQUIRED)

if(COSMA_BLAS_INTERFACE STREQUAL "32bits")
  set(_nvpl_int "_lp64")
else()
  set(_nvpl_int "_ilp64")
endif()

if(COSMA_BLAS_THREADING STREQUAL "openmp")
  set(_nvpl_thread "_omp")
else()
  set(_nvpl_thread "_seq")
endif()

if(NOT TARGET "cosma::BLAS::NVPL::scalapack_link")
  add_library("cosma::BLAS::NVPL::scalapack_link" INTERFACE IMPORTED)
  target_link_libraries("cosma::BLAS::NVPL::scalapack_link" INTERFACE
    "nvpl::blas${_nvpl_int}${_nvpl_thread}" "nvpl::lapack${_nvpl_int}${_nvpl_thread}")
endif()
