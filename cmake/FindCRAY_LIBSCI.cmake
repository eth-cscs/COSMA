include(FindPackageHandleStandardArgs)

# we are using the GNU compiler
set(_sciname "sci_gnu_mpi_mp")
set(_sciname_acc "sci_acc_gnu_nv60")

find_library(COSMA_CRAY_LIBSCI_LIBRARIES
    NAMES ${_sciname_acc} ${_sciname}
    HINTS
    ${_SCALAPACK_LIBRARY_DIRS}
    ENV CRAY_LIBSCI_PREFIX_DIR
    ENV CRAY_PE_LIBSCI_PREFIX_DIR
    ENV CRAY_LIBSCI_ACC_PREFIX_DIR
    ENV CRAY_PE_LIBSCI_ACC_PREFIX_DIR
    PATH_SUFFIXES lib
    DOC "Path to the Cray-libsci library.")

message("CRAY_LIBSCI: ${COSMA_CRAY_LIBSCI_LIBRARIES}")

find_package_handle_standard_args(CRAY_LIBSCI DEFAULT_MSG COSMA_CRAY_LIBSCI_LIBRARIES)

if (NOT TARGET cosma::BLAS::CRAY_LIBSCI::sci)
	add_library(cosma::BLAS::CRAY_LIBSCI::sci INTERFACE IMPORTED)
	set_target_properties(cosma::BLAS::CRAY_LIBSCI::sci PROPERTIES INTERFACE_LINK_LIBRARIES "${COSMA_CRAY_LIBSCI_LIBRARIES}")
	add_library(cosma::BLAS::CRAY_LIBSCI::blas ALIAS cosma::BLAS::CRAY_LIBSCI::sci)

	add_library(cosma::BLAS::CRAY_LIBSCI::scalapack_link INTERFACE IMPORTED)
	set_target_properties(cosma::BLAS::CRAY_LIBSCI::scalapack_link PROPERTIES INTERFACE_LINK_LIBRARIES "${COSMA_CRAY_LIBSCI_LIBRARIES}")
endif()
