include(FindPackageHandleStandardArgs)

if(COSMA_SCALAPACK STREQUAL "MKL")
	find_package(MKL REQUIRED)
  get_target_property(COSMA_SCALAPACK_LINK_LIBRARIES cosma::BLAS::MKL::scalapack_link
    INTERFACE_LINK_LIBRARIES)
elseif(COSMA_SCALAPACK STREQUAL "CRAY_LIBSCI")
	find_package(CRAY_LIBSCI REQUIRED)
	get_target_property(COSMA_SCALAPACK_LINK_LIBRARIES cosma::BLAS::CRAY_LIBSCI::scalapack_link
    INTERFACE_LINK_LIBRARIES)
elseif(COSMA_SCALAPACK STREQUAL "NVPL")
	find_package(nvpl REQUIRED)
	get_target_property(COSMA_SCALAPACK_LINK_LIBRARIES cosma::BLAS::NVPL::scalapack_link
    INTERFACE_LINK_LIBRARIES)
elseif(COSMA_SCALAPACK STREQUAL "CUSTOM")
  find_library(COSMA_SCALAPACK_LINK_LIBRARIES
    NAMES scalapack
    HINTS
    ${_COSMA_SCALAPACK_LIBRARY_DIRS}
    ENV SCALAPACKROOT
    ENV SCALAPACK_ROOT
    ENV ORNL_SCALAPACK_ROOT
    ENV SCALAPACK_PREFIX
    ENV SCALAPACK_DIR
    ENV SCALAPACKDIR
    /usr/bin
    PATH_SUFFIXES lib
    DOC "Path to the scalapack library.")
endif()

find_package_handle_standard_args(SCALAPACK REQUIRED_VARS COSMA_SCALAPACK_LINK_LIBRARIES)

set(COSMA_SCALAPACK_FOUND "YES")

if (NOT TARGET cosma::scalapack::scalapack)
  add_library(cosma::scalapack::scalapack INTERFACE IMPORTED)
endif()

set_target_properties(
  cosma::scalapack::scalapack PROPERTIES INTERFACE_LINK_LIBRARIES
  "${COSMA_SCALAPACK_LINK_LIBRARIES}")

mark_as_advanced(COSMA_SCALAPACK_LINK_LIBRARIES COSMA_SCALAPACK_FOUND)
