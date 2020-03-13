# find OpenBLAS
# workaround for missing openblas cmake config file in fedora

include(FindPackageHandleStandardArgs)

find_path(OPENBLAS_INCLUDE_DIR
  NAMES lapack.h
  PATH_SUFFIXES include
  HINTS
  ENV OPENBLAS_DIR
  ENV OPENBLASROOT
  DOC "openblas include directory")

find_library(OPENBLAS_LIBRARIES
  NAMES openblas
  PATH_SUFFIXES lib lib64
  HINTS
  ENV OPENBLAS_DIR
  ENV OPENBLASROOT
  DOC "openblas libraries list")

find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OPENBLAS_LIBRARIES OPENBLAS_INCLUDE_DIR)

if(OpenBLAS_FOUND AND NOT TARGET openblas)
  add_library(openblas INTERFACE IMPORTED)
  set_target_properties(openblas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${OPENBLAS_LIBRARIES}")
endif()
