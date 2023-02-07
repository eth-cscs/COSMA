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
#include(cp2k_utils)

find_package(PkgConfig)

cp2k_set_default_paths(ARMPL "Armpl")

foreach(_var armpl_ilp64 armpl_lp64 armpl_ilp64_mp armpl_lp64_mp)
  string(TOUPPER ${_var} _var_up)
  find_library("COSMA_${_var_up}_LINK_LIBRARIES" ${_var})
endforeach()

cp2k_include_dirs(ARMPL "armpl.h")

# Check for 64bit Integer support
if(COSMA_BLAS_INTERFACE MATCHES "64bits")
  set(COSMA_BLAS_armpl_LIB "armpl_ilp64")
else()
  set(COSMA_BLAS_armpl_LIB "armpl_lp64")
endif()

# Check for OpenMP support, VIA BLAS_VENDOR of Arm_mp or Arm_ipl64_mp
if(COSMA_BLAS_THREADING MATCHES "openmp")
  string(APPEND COSMA_BLAS_armpl_LIB "_mp")
endif()

# check if found
find_package_handle_standard_args(
  Armpl REQUIRED_VARS COSMA_ARMPL_INCLUDE_DIRS COSMA_ARMPL_LP64_LIBRARIES
  COSMA_ARMPL_LP64_MP_LIBRARIES COSMA_ARMPL_ILP64_LIBRARIES COSMA_ARMPL_ILP64_MP_LIBRARIES)

# add target to link against
if(COSMA_ARMPL_LP64_FOUND)

  if (NOT TARGET ARMPL::armpl)
    add_library(cosma::BLAS::ARMPL::armpl INTERFACE IMPORTED)
    # now define an alias to the target library
    add_library(cosma::BLAS::ARMPL::blas ALIAS cosma::BLAS::ARMPL::armpl)
  endif()

  # we need to iniitialize the targets of each individual libraries only once.
  if (NOT TARGET cosma::BLAS::ARMPL::${_var})
    foreach(_var armpl_ilp64 armpl_lp64 armpl_ilp64_mp armpl_lp64_mp)
      string(TOUPPER "COSMA_${_var}_LINK_LIBRARIES" _var_up)
        add_library(cosma::BLAS::ARMPL::${_var} INTERFACE IMPORTED)
        set_property(TARGET cosma::BLAS::ARMPL::${_var} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
          ${COSMA_ARMPL_INCLUDE_DIRS})
        set_property(TARGET cosma::BLAS::ARMPL::${_var} PROPERTY INTERFACE_LINK_LIBRARIES
          "${${_var_up}}")
      endif()
    endforeach()
  endif()

  set_property(TARGET cosma::BLAS::ARMPL::armpl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${COSMA_ARMPL_INCLUDE_DIRS})
  set_property(TARGET cosma::BLAS::ARMPL::armpl PROPERTY INTERFACE_LINK_LIBRARIES
    "COSMA_${COSMA_BLAS_armpl_LIB}_LINK_LIBRARIES")
endif()

mark_as_advanced(COSMA_ARMPL_FOUND COSMA_ARMPL_LINK_LIBRARIES
                 COSMA_ARMPL_INCLUDE_DIRS)
