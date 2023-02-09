# Copyright (c) 2022- ETH Zurich
#
# authors : Mathieu Taillefumier

include(FindPackageHandleStandardArgs)

set(_FLEXIBLAS_PATHS ${FLEXIBLAS_ROOT}
  $ENV{FLEXIBLAS_ROOT}
  $ENV{FLEXIBLASROOT}
  $ENV{FLEXIBLAS_DIR}
  $ENV{FLEXIBLASDIR}
  $ENV{ORNL_FLEXIBLAS_ROOT}
  $ENV{CRAY_FLEXIBLAS_ROOT})

# try first with pkg-config
find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(COSMA_FLEXIBLAS IMPORTED_TARGET GLOBAL flexiblas)
endif()

find_package_handle_standard_args(
  FLEXIBLAS DEFAULT_MSG COSMA_FLEXIBLAS_INCLUDE_DIRS
  COSMA_FLEXIBLAS_LINK_LIBRARIES)

if(COSMA_FLEXIBLAS_FOUND)
  set(COSMA_BLAS_VENDOR "FlexiBLAS")
  
  if(NOT TARGET cosma::BLAS::FLEXIBLAS::flexiblas)
    add_library(cosma::BLAS::FLEXIBLAS::flexiblas INTERFACE IMPORTED)
    add_library(cosma::BLAS::FLEXIBLAS::blas ALIAS cosma::BLAS::FLEXIBLAS::flexiblas)
  endif()
  set_target_properties(
    cosma::BLAS::FLEXIBLAS::flexiblas PROPERTIES INTERFACE_LINK_LIBRARIES
    "${COSMA_FLEXIBLAS_LINK_LIBRARIES}")
  if(COSMA_FLEXIBLAS_INCLUDE_DIRS)
    set_target_properties(
      cosma::BLAS::FLEXIBLAS::flexiblas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
      "${COSMA_FLEXIBLAS_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(COSMA_FLEXIBLAS_FOUND COSMA_FLEXIBLAS_INCLUDE_DIRS
                 COSMA_FLEXIBLAS_LINK_LIBRARIES COSMA_BLAS_VENDOR)
