# Copyright (c) 2022- ETH Zurich
#
# authors : Mathieu Taillefumier

if(NOT POLICY CMP0074)
  set(_GenericBLAS_PATHS ${GenericBLAS_ROOT} $ENV{GenericBLAS_ROOT})
endif()

find_library(
  COSMA_GenericBLAS_LINK_LIBRARIES
  NAMES "blas"
  HINTS ${_GenericBLAS_PATHS})
find_library(
  # optinally look for cblas library - not required
  COSMA_GenericBLAS_CBLAS_LIBRARIES
  NAMES "cblas"
  HINTS ${_GenericBLAS_PATHS})
find_path(
  COSMA_GenericBLAS_INCLUDE_DIRS
  NAMES "cblas.h"
  HINTS ${_GenericBLAS_PATHS})

# check if found
include(FindPackageHandleStandardArgs)
if(COSMA_GenericBLAS_INCLUDE_DIRS)
  find_package_handle_standard_args(
    GenericBLAS REQUIRED_VARS COSMA_GenericBLAS_INCLUDE_DIRS COSMA_GenericBLAS_LINK_LIBRARIES)
else()
  find_package_handle_standard_args(GenericBLAS
                                    REQUIRED_VARS COSMA_GenericBLAS_LINK_LIBRARIES)
endif()

if(COSMA_GenericBLAS_CBLAS_LINK_LIBRARIES)
  list(APPEND GenericBLAS_LINK_LIBRARIES ${GenericBLAS_CBLAS_LINK_LIBRARIES})
endif()

# add target to link against
if(NOT TARGET cosma::GenericBLAS::blas)
  add_library(cosma::GenericBLAS::blas INTERFACE IMPORTED)
endif()
set_property(TARGET cosma::GenericBLAS::blas PROPERTY INTERFACE_LINK_LIBRARIES
  ${COSMA_GenericBLAS_LINK_LIBRARIES})
set_property(
  TARGET cosma::GenericBLAS::blas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${COSMA_GenericBLAS_INCLUDE_DIRS})
endif()

# prevent clutter in cache
mark_as_advanced(COSMA_GenericBLAS_FOUND COSMA_GenericBLAS_LINK_LIBRARIES
                 COSMA_GenericBLAS_INCLUDE_DIRS COSMA_GenericBLAS_CBLAS_LIBRARIES)
