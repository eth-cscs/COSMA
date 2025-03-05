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

if(NOT TARGET "cosma::BLAS::NVPL::nvpl")
  add_library("cosma::BLAS::NVPL::nvpl" INTERFACE IMPORTED)
  target_link_libraries("cosma::BLAS::NVPL::nvpl" INTERFACE
    "nvpl::blas${_nvpl_int}${_nvpl_thread}" "nvpl::lapack${_nvpl_int}${_nvpl_thread}")
  get_target_property(COSMA_NVPL_BLAS_LIBRARIES "nvpl::blas${_nvpl_int}${_nvpl_thread}" nvpl_blas_LIBRARY_DIR)
  get_target_property(COSMA_NVPL_LAPACK_LIBRARIES "nvpl::lapack${_nvpl_int}${_nvpl_thread}" nvpl_lapack_LIBRARY_DIR)

  add_library(cosma::BLAS::NVPL::blas ALIAS cosma::BLAS::NVPL::nvpl)
  add_library(cosma::BLAS::NVPL::scalapack_link INTERFACE IMPORTED)
  set_target_properties(
    cosma::BLAS::NVPL::scalapack_link 
    PROPERTIES INTERFACE_LINK_LIBRARIES "${COSMA_NVPL_BLAS_LIBRARIES};${COSMA_NVPL_LAPACK_LIBRARIES}")
endif()
