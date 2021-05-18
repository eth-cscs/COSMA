include(FindPackageHandleStandardArgs)
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(_sciname "sci_gnu_mpi_mp")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(_sciname "sci_intel_mpi_mp")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(_sciname "sci_cray_mpi_mp")
else()
    message(${CMAKE_CXX_COMPILER_ID})
    message(FATAL_ERROR "Unknown compiler. When using cray-libsci use either GNU, INTEL or Clang compiler")
endif()
find_library(CRAY_LIBSCI_LIBRARIES
    NAMES ${_sciname_acc} ${_sciname}
    HINTS
    ${_SCALAPACK_LIBRARY_DIRS}
    ENV SCALAPACKROOT
    ENV CRAY_LIBSCI_PREFIX_DIR
    ENV CRAY_LIBSCI_PREFIX_DIR
    ENV CRAY_LIBSCI_ACC_PREFIX_DIR
    PATH_SUFFIXES "lib" "lib64"
    DOC "Path to the Cray-libsci library")

message("CRAY_LIBSCI: ${CRAY_LIBSCI_LIBRARIES}")
find_package_handle_standard_args(CRAY_LIBSCI DEFAULT_MSG CRAY_LIBSCI_LIBRARIES)
