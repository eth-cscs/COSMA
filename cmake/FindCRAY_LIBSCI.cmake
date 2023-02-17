include(FindPackageHandleStandardArgs)

# we are using the GNU compiler
set(_sciname "sci_gnu_mpi_mp")
set(_sciname_acc "sci_acc_gnu_nv60")

find_library(CRAY_LIBSCI_LINK_LIBRARIES
    NAMES ${_sciname_acc} ${_sciname}
    HINTS
    ${_SCALAPACK_LIBRARY_DIRS}
    ENV CRAY_LIBSCI_PREFIX_DIR
    ENV CRAY_LIBSCI_ACC_PREFIX_DIR
    PATH_SUFFIXES lib
    DOC "Path to the Cray-libsci library.")

message("CRAY_LIBSCI: ${CRAY_LIBSCI_LIBRARIES}")

find_package_handle_standard_args(CRAY_LIBSCI DEFAULT_MSG CRAY_LIBSCI_LIBRARIES)

if (CRAY_LIBSCI_LIBRARIES AND NOT TARGET cosma::BLAS::SCI::scalapack)
  add_library(cosma::BLAS::SCI::sci INTERFACE IMPORTED)
  add_library(cosma::BLAS::SCI::blas ALIAS cosma::BLAS::SCI::sci)
  add_library(cosma::BLAS::SCI::scalapack_link INTERFACE IMPORTED)
  set_properties(cosma::BLAS::SCI::scalapack PROPERTY INTERFACE_LINK_LIBRARIES "${CRAY_LIBSCI_LINK_LIBRARIES}")
  set_properties(cosma::BLAS::SCI::sci PROPERTY INTERFACE_LINK_LIBRARIES "${CRAY_LIBSCI_LINK_LIBRARIES}")
endif()
