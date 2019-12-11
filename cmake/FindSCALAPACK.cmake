include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_SCALAPACK scalapack)
find_library(SCALAPACK_LIBRARIES
    NAMES scalapack ${_sciname}
    HINTS
    ${_SCALAPACK_LIBRARY_DIRS}
    ENV SCALAPACKROOT
    ENV SCALAPACK_ROOT
    ENV SCALAPACK_DIR
    ENV SCALAPACKDIR
    ENV /usr/bin
    PATH_SUFFIXES lib
    DOC "Path to the scalapack library.")

find_package_handle_standard_args(SCALAPACK DEFAULT_MSG SCALAPACK_LIBRARIES)
