# Copyright (c) 2022- ETH Zurich
#
# authors : Mathieu Taillefumier

include(FindPackageHandleStandardArgs)

set(_ARMPL_PATHS ${ARMPL_ROOT}
  $ENV{ARMPL_ROOT}
  $ENV{ARMPLROOT}
  $ENV{ARMPL_DIR}
  $ENV{ARMPLDIR}
  $ENV{ORNL_ARMPL_ROOT}
  $ENV{CRAY_ARMPL_ROOT})

foreach(_var armpl armpl_int64 armpl_ilp64 armpl_lp64 armpl_ilp64_mp armpl_lp64_mp)
  string(TOUPPER ${_var} _var_up)
  find_library("COSMA_${_var_up}_LINK_LIBRARIES" NAME ${_var} HINTS ${_ARMPL_PATHS} PATH_SUFFIXES "lib" "lib64" "armpl/lib" "armpl/lib64" "armpl")
endforeach()

find_path(COSMA_ARMPL_INCLUDE_DIRS NAMES "armpl.h" HINTS ${_ARMPL_PATHS} PATH_SUFFIXES "include" "armpl" "armpl/include" "include/armpl")

# Check for 64bit Integer support
if(COSMA_BLAS_INTERFACE MATCHES "64bits")
  set(COSMA_BLAS_armpl_LIB "ARMPL_ILP64")
else()
  set(COSMA_BLAS_armpl_LIB "ARMPL_LP64")
endif()

# Check for OpenMP support, VIA BLAS_VENDOR of Arm_mp or Arm_ipl64_mp
if(COSMA_BLAS_THREADING MATCHES "openmp")
  string(APPEND COSMA_BLAS_armpl_LIB "_MP")
endif()

# check if found
find_package_handle_standard_args(
  Armpl REQUIRED_VARS COSMA_ARMPL_INCLUDE_DIRS COSMA_ARMPL_LP64_LINK_LIBRARIES
  COSMA_ARMPL_LP64_MP_LINK_LIBRARIES COSMA_ARMPL_ILP64_LINK_LIBRARIES COSMA_ARMPL_ILP64_MP_LINK_LIBRARIES)

# add target to link against
if (NOT TARGET cosma::BLAS::ARMPL::armpl)
  add_library(cosma::BLAS::ARMPL::armpl INTERFACE IMPORTED)
  # now define an alias to the target library
  add_library(cosma::BLAS::ARMPL::blas ALIAS cosma::BLAS::ARMPL::armpl)
endif()

# we need to iniitialize the targets of each individual libraries only once.
foreach(_var armpl_ilp64 armpl_lp64 armpl_ilp64_mp armpl_lp64_mp)
  string(TOUPPER "${_var}" _var_up)
  if (NOT TARGET cosma::BLAS::ARMPL::${_var})
    add_library(cosma::BLAS::ARMPL::${_var} INTERFACE IMPORTED)
    set_property(TARGET cosma::BLAS::ARMPL::${_var} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${COSMA_ARMPL_INCLUDE_DIRS})
    set_property(TARGET cosma::BLAS::ARMPL::${_var} PROPERTY INTERFACE_LINK_LIBRARIES
      "${COSMA_${_var_up}_LINK_LIBRARIES}")
  endif()
endforeach()

set_property(TARGET cosma::BLAS::ARMPL::armpl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  ${COSMA_ARMPL_INCLUDE_DIRS})
set_property(TARGET cosma::BLAS::ARMPL::armpl PROPERTY INTERFACE_LINK_LIBRARIES
  "${COSMA_${COSMA_BLAS_armpl_LIB}_LINK_LIBRARIES}")
endif()

set(COSMA_BLAS_VENDOR "ARMPL")

mark_as_advanced(COSMA_ARMPL_FOUND COSMA_BLAS_VENDOR COSMA_ARMPL_INCLUDE_DIRS)
