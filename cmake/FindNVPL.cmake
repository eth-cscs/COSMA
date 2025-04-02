find_package("nvpl_blas" REQUIRED)
find_package("nvpl_lapack" REQUIRED)
find_package("nvpl_scalapack" REQUIRED)

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

if("${MPI_CXX_LIBRARY_VERSION_STRING}" MATCHES "Open MPI")
  if(MPI_VERSION VERSION_GREATER_EQUAL "5.0")
    set(_nvpl_mpi "_openmpi5")
  elseif(MPI_VERSION VERSION_GREATER_EQUAL "4.0")
    set(_nvpl_mpi "_openmpi4")
  else(MPI_VERSION VERSION_GREATER_EQUAL "3.0")
    set(_nvpl_mpi "_openmpi3")
  endif()
else()
  set(_nvpl_mpi "_mpich")
endif()

if(NOT TARGET "cosma::BLAS::NVPL::nvpl")
  add_library("cosma::BLAS::NVPL::nvpl" INTERFACE IMPORTED)
  target_link_libraries("cosma::BLAS::NVPL::nvpl" INTERFACE
    "nvpl::blas${_nvpl_int}${_nvpl_thread}" "nvpl::lapack${_nvpl_int}${_nvpl_thread}"
    "nvpl::blacs${_nvpl_int}${_nvpl_mpi}" "nvpl::scalapack${_nvpl_int}")

  get_target_property(COSMA_NVPL_LAPACK_LIBRARIES "nvpl::lapack${_nvpl_int}${_nvpl_thread}" INTERFACE_LINK_LIBRARIES)
  get_target_property(COSMA_NVPL_SCALAPACK_LIBRARIES "nvpl::scalapack${_nvpl_int}" INTERFACE_LINK_LIBRARIES)
  get_target_property(COSMA_NVPL_BLAS_INCLUDE_DIRS "nvpl::blas${_nvpl_int}${_nvpl_thread}" INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(COSMA_NVPL_LAPACK_INCLUDE_DIRS "nvpl::lapack${_nvpl_int}${_nvpl_thread}" INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(COSMA_NVPL_SCALAPACK_INCLUDE_DIRS "nvpl::scalapack${_nvpl_int}" INTERFACE_INCLUDE_DIRECTORIES)

  set_target_properties(
    cosma::BLAS::NVPL::nvpl 
    PROPERTIES INTERFACE_LINK_LIBRARIES 
    "${COSMA_NVPL_LAPACK_LIBRARIES}")
  set_target_properties(
    cosma::BLAS::NVPL::nvpl
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES 
    "${COSMA_NVPL_BLAS_INCLUDE_DIRS};${COSMA_NVPL_LAPACK_INCLUDE_DIRS}")

  add_library(cosma::BLAS::NVPL::blas ALIAS cosma::BLAS::NVPL::nvpl)

  add_library(cosma::BLAS::NVPL::scalapack_link INTERFACE IMPORTED)
  set_target_properties(
    cosma::BLAS::NVPL::scalapack_link 
    PROPERTIES INTERFACE_LINK_LIBRARIES 
    "${COSMA_NVPL_LAPACK_LIBRARIES};${COSMA_NVPL_SCALAPACK_LIBRARIES}")
  set_target_properties(
    cosma::BLAS::NVPL::scalapack_link 
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES 
    "${COSMA_NVPL_BLAS_INCLUDE_DIRS};${COSMA_NVPL_LAPACK_INCLUDE_DIRS};${COSMA_NVPL_SCALAPACK_INCLUDE_DIRS}")
endif()
