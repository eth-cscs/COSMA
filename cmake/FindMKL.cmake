# Uses MKLROOT environment variable or CMake's MKL_ROOT to find MKL.
#
# Imported Targets: 
#   MKL::MKL
#
# The type of threading for MKL has to be specified using the variable
#        MKL_THREADING            := IOMP|GOMP           (default: serial)
#        MKL_USE_64BIT_INTEGERS   := True|False          (default: False)
#        MKL_MPI_TYPE             := OMPI|MPICH          (default: no ScaLAPACK)
#
# NOT SUPPORTED
#   - TBB threading back-end
#   - F95 interfaces
#
# Note: Do not mix GCC and Intel OpenMP.
#       Do not mix MPI implementations.
#       The module depends on FindThreads, FindOpenMP and FindMPI (if ScaLAPACK found)
#
include(FindPackageHandleStandardArgs)

find_path(MKL_INCLUDE_DIR mkl.h
    HINTS
        $ENV{MKLROOT}/include
        ${MKL_ROOT}/include
)
mark_as_advanced(MKL_INCLUDE_DIR)

set(_mkl_libpath_suffix "lib/intel64")
if(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32 bit
    set(_mkl_libpath_suffix "lib/ia32")
endif()

if (WIN32)
    string(APPEND _mkl_libpath_suffix "_win")
elseif (APPLE)
    string(APPEND _mkl_libpath_suffix "_mac")
else ()
    string(APPEND _mkl_libpath_suffix "_lin")
endif ()

function(__mkl_find_library _name)
    find_library(${_name}
        NAMES ${ARGN}
        HINTS ENV MKLROOT
              ${MKL_ROOT}
        PATH_SUFFIXES ${_mkl_libpath_suffix}
    )
    mark_as_advanced(${_name})
endfunction()

__mkl_find_library(MKL_CORE_LIB mkl_core)

if(NOT MKL_THREADING)
  __mkl_find_library(MKL_THREADING_LIB mkl_sequential)
elseif(MKL_THREADING MATCHES "GOMP")
  __mkl_find_library(MKL_THREADING_LIB mkl_gnu_thread)
elseif(MKL_THREADING MATCHES "IOMP")
  __mkl_find_library(MKL_THREADING_LIB mkl_intel_thread)
endif()

if(NOT MKL_USE_64BIT_INTEGERS)
  __mkl_find_library(MKL_INTERFACE_LIB mkl_intel_lp64)
else()
  __mkl_find_library(MKL_INTERFACE_LIB mkl_intel_ilp64)
endif()

find_package_handle_standard_args(MKL 
  DEFAULT_MSG  MKL_CORE_LIB
               MKL_THREADING_LIB
               MKL_INTERFACE_LIB
               MKL_INCLUDE_DIR
  )

# ScaLAPACK
# 
if(MKL_MPI_TYPE)
  __mkl_find_library(MKL_SCALAPACK_LIB mkl_scalapack_lp64)
  if (MKL_MPI_TYPE MATCHES "MPICH")
    __mkl_find_library(MKL_BLACS_LIB mkl_blacs_intelmpi_lp64)
  elseif(MKL_MPI_TYPE MATCHES "OMPI")
    __mkl_find_library(MKL_BLACS_LIB mkl_blacs_openmpi_lp64)
  endif()

  find_package_handle_standard_args(MKL_SCALAPACK 
    DEFAULT_MSG MKL_BLACS_LIB
                MKL_SCALAPACK_LIB
  )
endif()

if (MKL_FOUND AND NOT TARGET MKL::MKL)
    find_package(Threads REQUIRED)
    add_library(MKL::CORE UNKNOWN IMPORTED)
    set_target_properties(MKL::CORE PROPERTIES 
      IMPORTED_LOCATION ${MKL_CORE_LIB}
      INTERFACE_LINK_LIBRARIES Threads::Threads)

    if(MKL_THREADING)
      find_package(OpenMP REQUIRED)
      set_target_properties(MKL::CORE PROPERTIES 
        INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    endif()

    if(MKL_MPI_TYPE)
      add_library(MKL::BLACS UNKNOWN IMPORTED)
      set_target_properties(MKL::BLACS PROPERTIES 
        IMPORTED_LOCATION ${MKL_BLACS_LIB})
      set_target_properties(MKL::CORE PROPERTIES 
        INTERFACE_LINK_LIBRARIES MKL::BLACS)
    endif()

    add_library(MKL::THREADING UNKNOWN IMPORTED)
    set_target_properties(MKL::THREADING PROPERTIES 
        IMPORTED_LOCATION ${MKL_THREADING_LIB}
        INTERFACE_LINK_LIBRARIES MKL::CORE)

    add_library(MKL::BLAS_INTERFACE UNKNOWN IMPORTED)
    set_target_properties(MKL::BLAS_INTERFACE PROPERTIES 
      IMPORTED_LOCATION ${MKL_INTERFACE_LIB}
      INTERFACE_LINK_LIBRARIES MKL::THREADING)

    # The MKL::MKL target
    #
    add_library(MKL::MKL INTERFACE IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES 
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES MKL::BLAS_INTERFACE)

    if(MKL_MPI_TYPE)
      add_library(MKL::SCALAPACK UNKNOWN IMPORTED)
      set_target_properties(MKL::SCALAPACK PROPERTIES 
        IMPORTED_LOCATION ${MKL_SCALAPACK_LIB}
        INTERFACE_LINK_LIBRARIES MKL::BLAS_INTERFACE)

      set_target_properties(MKL::MKL PROPERTIES 
        INTERFACE_LINK_LIBRARIES MKL::SCALAPACK)
    else()
      set_target_properties(MKL::MKL PROPERTIES 
        INTERFACE_LINK_LIBRARIES MKL::BLAS_INTERFACE)
    endif()
endif()

