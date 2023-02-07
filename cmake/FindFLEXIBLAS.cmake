#!-------------------------------------------------------------------------------------------------!
#!   CP2K: A general program to perform molecular dynamics simulations                             !
#!   Copyright 2000-2023 CP2K developers group <https://cp2k.org>                                  !
#!                                                                                                 !
#!   SPDX-License-Identifier: GPL-2.0-or-later                                                     !
#!-------------------------------------------------------------------------------------------------!

# Copyright (c) 2022- ETH Zurich
#
# authors : Mathieu Taillefumier

include(FindPackageHandleStandardArgs)
include(cp2k_utils)

cp2k_set_default_paths(FLEXIBLAS "FlexiBLAS")

# try first with pkg-config
find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(COSMA_FLEXIBLAS IMPORTED_TARGET GLOBAL flexiblas)
endif()

# manual; search
if(NOT COSMA_FLEXIBLAS_FOUND)
  find_path(
    COSMA_FLEXIBLAS_INCLUDE_DIRS
    NAMES cblas.h
    PATHS "${FLEXIBLAS_ROOT}"
    HINTS "${FLEXIBLAS_ROOT}"
    ENV ${FLEXIBLAS_ROOT}
    ENV ${ORNL_FLEXIBLAS_ROOT}
    PATH_SUFFIXES "include" "include/${_pacakge_name}" "${_package_name}")
endif()

# search for include directories anyway
if(NOT COSMA_FLEXIBLAS_INCLUDE_DIRS)
  cp2k_include_dirs(FLEXIBLAS "flexiblas.h")
endif()

find_package_handle_standard_args(
  FlexiBLAS DEFAULT_MSG COSMA_FLEXIBLAS_INCLUDE_DIRS
  COSMA_FLEXIBLAS_LINK_LIBRARIES)

if(NOT COSMA_FLEXIBLAS_FOUND)
  set(COSMA_BLAS_VENDOR "FlexiBLAS")
endif()

if(COSMA_FLEXIBLAS_FOUND)
  if(NOT TARGET COSMA_FlexiBLAS::flexiblas)
    add_library(CP2K::BLAS::FlexiBLAS::flexiblas INTERFACE IMPORTED)
    add_library(CP2K::BLAS::FlexiBLAS::blas ALIAS CP2K::BLAS::FlexiBLAS::flexiblas)
  endif()
  set_target_properties(
    CP2K::BLAS::FlexiBLAS::flexiblas PROPERTIES INTERFACE_LINK_LIBRARIES
                                         "${COSMA_FLEXIBLAS_LINK_LIBRARIES}")
  if(COSMA_FLEXIBLAS_INCLUDE_DIRS)
    set_target_properties(
      CP2K::BLAS::FlexiBLAS::flexiblas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                           "${COSMA_FLEXIBLAS_INCLUDE_DIRS}")
  endif()
  set(COSMA_BLAS_VENDOR "FlexiBLAS")
endif()

mark_as_advanced(COSMA_FLEXIBLAS_FOUND COSMA_FLEXIBLAS_INCLUDE_DIRS
                 COSMA_FLEXIBLAS_LINK_LIBRARIES COSMA_BLAS_VENDOR)
