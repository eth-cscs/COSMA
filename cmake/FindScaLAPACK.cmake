# Uses MKLROOT environment variable or CMake's MKL_ROOT to find MKL ScaLAPACK.
#
# Variables:
#     ScaLAPACK_FOUND
#
# Imported Targets: 
#     MKL::ScaLAPACK
#
# Note: The default is MPICH. Do not mix MPI implementations.
#
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(FindPackageHandleStandardArgs)

find_package(MPI REQUIRED)
find_package(MKL REQUIRED)

if (NOT COSMA_WITH_OPENMPI)
    include(${CMAKE_CURRENT_LIST_DIR}/check_for_openmpi.cmake)
    check_for_openmpi()
endif()

function(__mkl_find_library _name)
    find_library(${_name}
        NAMES ${ARGN}
        HINTS ${MKL_ROOT}
        PATH_SUFFIXES ${_mkl_libpath_suffix}
                      lib
    )
    mark_as_advanced(${_name})
endfunction()

__mkl_find_library(MKL_SCALAPACK_LIB mkl_scalapack_${_mkl_lp})

if(COSMA_WITH_OPENMPI) # OpenMPI
    if(APPLE)
        message(FATAL_ERROR "Only MPICH is supported on Apple.")
    endif()
     __mkl_find_library(MKL_BLACS_LIB mkl_blacs_openmpi_${_mkl_lp})
else() # MPICH
    if(APPLE)
        __mkl_find_library(MKL_BLACS_LIB mkl_blacs_mpich_${_mkl_lp})
    else()
        __mkl_find_library(MKL_BLACS_LIB mkl_blacs_intelmpi_${_mkl_lp})
    endif()
endif()

find_package_handle_standard_args(ScaLAPACK
  DEFAULT_MSG MKL_BLACS_LIB
              MKL_SCALAPACK_LIB
)

if (ScaLAPACK_FOUND AND NOT TARGET MKL::ScaLAPACK)
    add_library(MKL::BLACS UNKNOWN IMPORTED)
    set_target_properties(MKL::BLACS PROPERTIES IMPORTED_LOCATION ${MKL_BLACS_LIB})

    add_library(MKL::ScaLAPACK UNKNOWN IMPORTED)
    set_target_properties(MKL::ScaLAPACK PROPERTIES 
      IMPORTED_LOCATION "${MKL_SCALAPACK_LIB}"
      INTERFACE_LINK_LIBRARIES "MKL::BLAS_INTERFACE;MKL::THREADING;MKL::CORE;MKL::BLACS;${_mkl_threading_backend};Threads::Threads"
      )
endif()

