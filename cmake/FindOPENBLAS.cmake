# find OPENBLAS

include(FindPackageHandleStandardArgs)

# if(NOT POLICY CMP0074)
set(_OPENBLAS_PATHS ${OPENBLAS_ROOT} 
    $ENV{OPENBLAS_ROOT} 
    $ENV{OPENBLASROOT}
    $ENV{OPENBLAS_DIR}
    $ENV{OPENBLASDIR})
# endif()

find_path(COSMA_OPENBLAS_INCLUDE_DIRS
    NAMES "cblas-openblas.h" "cblas_openblas.h" "cblas.h"
    PATH_SUFFIXES "openblas" "openblas/include" "include" "include/openblas"
    HINTS ${_OPENBLAS_PATHS}
    DOC "openblas include directory")

  find_library(COSMA_OPENBLAS_LINK_LIBRARIES
    NAMES openblas
    PATH_SUFFIXES "lib" "lib64" "openblas/lib" "openblas/lib64" "openblas"
    HINTS ${_OPENBLAS_PATHS}
    DOC "openblas libraries list")

  find_package_handle_standard_args(OPENBLAS 
    DEFAULT_MSG 
    COSMA_OPENBLAS_LINK_LIBRARIES COSMA_OPENBLAS_INCLUDE_DIRS)
  
  if(NOT TARGET cosma::BLAS::OPENBLAS::openblas)
    add_library(cosma::BLAS::OPENBLAS::openblas INTERFACE IMPORTED)
    add_library(cosma::BLAS::OPENBLAS::blas ALIAS cosma::BLAS::OPENBLAS::openblas)
  endif()
  set_property(TARGET cosma::BLAS::OPENBLAS::openblas 
    PROPERTY INTERFACE_LINK_LIBRARIES ${COSMA_OPENBLAS_LIBRARIES})
  set_property(TARGET cosma::BLAS::OPENBLAS::openblas 
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${COSMA_OPENBLAS_INCLUDE_DIRS})

# prevent clutter in cache
MARK_AS_ADVANCED(OPENBLAS_FOUND OPENBLAS_LIBRARIES OPENBLAS_INCLUDE_DIRS)
