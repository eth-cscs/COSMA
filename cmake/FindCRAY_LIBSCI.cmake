include(FindPackageHandleStandardArgs)

# we are using the GNU compiler
set(_sciname "sci_gnu_mpi_mp")

find_library(CRAY_LIBSCI_LIBRARIES
    NAMES ${_sciname}
    HINTS
    ${_SCALAPACK_LIBRARY_DIRS}
    ENV CRAY_LIBSCI_PREFIX_DIR
    ENV CRAY_LIBSCI_DIR
    PATH_SUFFIXES lib
    DOC "Path to the Cray-libsci library.")

find_package_handle_standard_args(CRAY_LIBSCI DEFAULT_MSG CRAY_LIBSCI_LIBRARIES)
