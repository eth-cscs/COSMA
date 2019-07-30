# Uses MKLROOT environment variable or CMake's MKL_ROOT to find MKL.
#
# Imported Targets: 
#   MKL::MKL
#
# The type of threading for MKL has to be specified using the variable
#        MKL_PARALLEL            := ON|OFF   (default: ON / parallel)
#        MKL_64BIT               := ON|OFF   (default: OFF / 32bit interface)
#
# NOT SUPPORTED
#   - TBB threading back-end
#   - F95 interfaces
#
# Note: Mixing GCC and Intel OpenMP backends is a bad idea.
#       The module depends on FindThreads and FindOpenMP.
#
include(FindPackageHandleStandardArgs)

set(MKL_ROOT "$ENV{MKLROOT}" CACHE FILEPATH "MKL's ROOT directory.")

option(MKL_PARALLEL "Parallel version of MKL" ON)
option(MKL_64BIT "Parallel version of MKL" OFF)

find_path(MKL_INCLUDE_DIR mkl.h
    HINTS ${MKL_ROOT}/include
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
        HINTS ${MKL_ROOT}
        PATH_SUFFIXES ${_mkl_libpath_suffix}
                      lib
        )
    mark_as_advanced(${_name})
endfunction()

__mkl_find_library(MKL_CORE_LIB mkl_core)

set(_mkl_lp "lp64")
mark_as_advanced(${_mkl_lp})
if(MKL_64BIT)
  set(_mkl_lp "ilp64")
endif()
__mkl_find_library(MKL_INTERFACE_LIB mkl_intel_${_mkl_lp})

if(MKL_PARALLEL)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
        __mkl_find_library(MKL_THREADING_LIB mkl_gnu_thread)
    else()
        __mkl_find_library(MKL_THREADING_LIB mkl_intel_thread)
    endif()
else()
    __mkl_find_library(MKL_THREADING_LIB mkl_sequential)
endif()

find_package_handle_standard_args(MKL 
    DEFAULT_MSG  MKL_CORE_LIB
    MKL_THREADING_LIB
    MKL_INTERFACE_LIB
    MKL_INCLUDE_DIR
    )

if (MKL_FOUND AND NOT TARGET MKL::MKL)
    find_package(Threads REQUIRED)

    set(_mkl_threading_backend "")
    if(MKL_THREADING)
        find_package(OpenMP REQUIRED)
        set(_mkl_threading_backend "OpenMP::OpenMP_CXX")
    endif()

    add_library(MKL::CORE UNKNOWN IMPORTED)
    set_target_properties(MKL::CORE PROPERTIES IMPORTED_LOCATION ${MKL_CORE_LIB})

    add_library(MKL::THREADING UNKNOWN IMPORTED)
    set_target_properties(MKL::THREADING PROPERTIES IMPORTED_LOCATION ${MKL_THREADING_LIB})

    add_library(MKL::BLAS_INTERFACE UNKNOWN IMPORTED)
    set_target_properties(MKL::BLAS_INTERFACE PROPERTIES IMPORTED_LOCATION ${MKL_INTERFACE_LIB})

    add_library(MKL::MKL INTERFACE IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "MKL::BLAS_INTERFACE;MKL::THREADING;MKL::CORE;${_mkl_threading_backend};Threads::Threads")
endif()

