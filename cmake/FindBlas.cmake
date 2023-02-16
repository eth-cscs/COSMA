# Copyright (c) 2022- ETH Zurich
#
# authors : Mathieu Taillefumier



if(NOT
   (CMAKE_C_COMPILER_LOADED
    OR CMAKE_CXX_COMPILER_LOADED
    OR CMAKE_Fortran_COMPILER_LOADED))
  message(FATAL_ERROR "FindBLAS requires Fortran, C, or C++ to be enabled.")
endif()

set(COSMA_BLAS_VENDOR_LIST
  "auto"
  "MKL"
  "OPENBLAS"
  "FLEXIBLAS"
  "ARMPL"
  "GenericBLAS"
  "CRAY_LIBSCI"
  "BLIS"
  "ATLAS"
  "OFF")

# COSMA_BLAS_VENDOR should normally be defined here but cosma defines it in the
# main CMakeLists.txt to keep the old behavior. the threading and integer
# interface can also be controlled but are fixed to the default values that
# COSMA was configured before introducing this module. So if findBLAS.cmake is
# to be used elsewhere, it is better to look at what CP2K does and start from
# there

if(NOT ${COSMA_BLAS_VENDOR} IN_LIST COSMA_BLAS_VENDOR_LIST)
  message(FATAL_ERROR "Invalid Host BLAS backend")
endif()

set(COSMA_BLAS_THREAD_LIST "sequential" "thread" "gnu-thread" "intel-thread"
  "tbb-thread" "openmp")

set(COSMA_BLAS_THREADING
  "openmp"
  CACHE STRING "threaded blas library")
set_property(CACHE COSMA_BLAS_THREADING PROPERTY STRINGS
  ${COSMA_BLAS_THREAD_LIST})

if(NOT ${COSMA_BLAS_THREADING} IN_LIST COSMA_BLAS_THREAD_LIST)
  message(FATAL_ERROR "Invalid threaded BLAS backend")
endif()

set(COSMA_BLAS_INTERFACE_BITS_LIST "32bits" "64bits")
set(COSMA_BLAS_INTERFACE
  "32bits"
  CACHE STRING
  "32 bits integers are used for indices, matrices and vectors sizes")
set_property(CACHE COSMA_BLAS_INTERFACE
  PROPERTY STRINGS ${COSMA_BLAS_INTERFACE_BITS_LIST})

if(NOT ${COSMA_BLAS_INTERFACE} IN_LIST COSMA_BLAS_INTERFACE_BITS_LIST)
  message(
    FATAL_ERROR
    "Invalid parameters. Blas and lapack can exist in two flavors 32 or 64 bits interfaces (relevant mostly for mkl)"
  )
endif()

if (COSMA_BLAS_VENDOR MATCHES "OFF")
   return ()
endif()

set(COSMA_BLAS_FOUND FALSE)

# first check for a specific implementation if requested

if(NOT COSMA_BLAS_VENDOR MATCHES "auto")
   if (COSMA_BLAS_VENDOR MATCHES "CUSTOM")
       find_package(GenericBLAS REQUIRED)
   else()
       find_package(${COSMA_BLAS_VENDOR} REQUIRED)
  endif()
  if(TARGET cosma::BLAS::${COSMA_BLAS_VENDOR}::blas)
    get_target_property(COSMA_BLAS_INCLUDE_DIRS cosma::BLAS::${COSMA_BLAS_VENDOR}::blas
                        INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(COSMA_BLAS_LINK_LIBRARIES cosma::BLAS::${COSMA_BLAS_VENDOR}::blas
                        INTERFACE_LINK_LIBRARIES)
    set(COSMA_BLAS_FOUND TRUE)
  endif()
else()
  # search for any blas implementation and exit imediately if one is found
  foreach(_libs ${COSMA_BLAS_VENDOR_LIST})
    # i exclude the first item of the list
    if (NOT _libs STREQUAL "auto")
      find_package(${_libs})
      if(TARGET cosma::BLAS::${_libs}::blas)
        get_target_property(COSMA_BLAS_INCLUDE_DIRS cosma::BLAS::${_libs}::blas
          INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(COSMA_BLAS_LINK_LIBRARIES cosma::BLAS::${_libs}::blas
          INTERFACE_LINK_LIBRARIES)
        set(COSMA_BLAS_VENDOR "${_libs}")
        set(COSMA_BLAS_FOUND TRUE)
        break()
      endif()
    endif()
  endforeach()
endif()

if(COSMA_BLAS_INCLUDE_DIRS)
  find_package_handle_standard_args(
    Blas REQUIRED_VARS COSMA_BLAS_LINK_LIBRARIES COSMA_BLAS_INCLUDE_DIRS
                       COSMA_BLAS_VENDOR)
else()
  find_package_handle_standard_args(Blas REQUIRED_VARS COSMA_BLAS_LINK_LIBRARIES
                                                       COSMA_BLAS_VENDOR)
endif()

if(NOT TARGET cosma::BLAS::blas)
  add_library(cosma::BLAS::blas INTERFACE IMPORTED)
endif()

set_target_properties(cosma::BLAS::blas PROPERTIES INTERFACE_LINK_LIBRARIES
  "${COSMA_BLAS_LINK_LIBRARIES}")

if(COSMA_BLAS_INCLUDE_DIRS)
  set_target_properties(cosma::BLAS::blas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    "${COSMA_BLAS_INCLUDE_DIRS}")
endif()

set(COSMA_BLAS ${COSMA_BLAS_VENDOR})

mark_as_advanced(COSMA_BLAS_INCLUDE_DIRS)
mark_as_advanced(COSMA_BLAS_LINK_LIBRARIES)
mark_as_advanced(COSMA_BLAS)
mark_as_advanced(COSMA_BLAS_FOUND)
