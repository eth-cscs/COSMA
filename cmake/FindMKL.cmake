#[=======================================================================[.rst:
FindMKL
-------

The following conventions are used:

seq / SEQ      - sequential MKL
omp / OMP      - threaded MKL with OpenMP back-end
32bit / 32BIT  - MKL 32 bit integer interface (used most often)
64bit / 64BIT  - MKL 64 bit integer interface

The module attempts to define a target for each MKL configuration. The 
configuration will not be available if there are missing library files or a 
missing dependency. Configurations can be requested explicitly as COMPONENTS:

  find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_SEQ BLAS_64BIT_OMP)

MKL is considered found if
  a. all required components have been found (if any). 
  b. mkl core libraries and headers have been found.

Example usage
^^^^^^^^^^^^^

  find_package(MKL)
  target_link_libraries(... mkl::blas_32bit_seq)

  or

  target_link_libraries(... mkl::blas_32bit_omp)

  or 

  target_link_libraries(... mkl::blas_64bit_omp)

  or

  target_link_libraries(... mkl::scalapack_32bit_omp)

Search variables
^^^^^^^^^^^^^^^^

``MKLROOT``
  Environment variable set to MKL's root directory  

``MKL_ROOT``
  CMake variable set to MKL's root directory  


Imported targets 
^^^^^^^^^^^^^^^^

mkl::blas_32bit_seq
mkl::blas_32bit_omp
mkl::blas_64bit_seq
mkl::blas_64bit_omp

mkl::blacs_32bit_seq
mkl::blacs_32bit_omp
mkl::blacs_64bit_seq
mkl::blacs_64bit_omp

mkl::scalapack_32bit_seq
mkl::scalapack_32bit_omp
mkl::scalapack_64bit_seq
mkl::scalapack_64bit_omp

Result variables
^^^^^^^^^^^^^^^^

MKL_FOUND

MKL_BLAS_32BIT_SEQ_FOUND
MKL_BLAS_32BIT_OMP_FOUND
MKL_BLAS_64BIT_SEQ_FOUND
MKL_BLAS_64BIT_OMP_FOUND

MKL_BLACS_32BIT_SEQ_FOUND
MKL_BLACS_32BIT_OMP_FOUND
MKL_BLACS_64BIT_SEQ_FOUND
MKL_BLACS_64BIT_OMP_FOUND

MKL_SCALAPACK_32BIT_SEQ_FOUND
MKL_SCALAPACK_32BIT_OMP_FOUND
MKL_SCALAPACK_64BIT_SEQ_FOUND
MKL_SCALAPACK_64BIT_OMP_FOUND

Components
^^^^^^^^^^

BLAS_32BIT_SEQ
BLAS_32BIT_OMP
BLAS_64BIT_SEQ
BLAS_64BIT_OMP

BLACS_32BIT_SEQ
BLACS_32BIT_OMP
BLACS_64BIT_SEQ
BLACS_64BIT_OMP

SCALAPACK_32BIT_SEQ
SCALAPACK_32BIT_OMP
SCALAPACK_64BIT_SEQ
SCALAPACK_64BIT_OMP

Not supported
^^^^^^^^^^^^^

- TBB threading back-end
- F95 interfaces

Note: Mixing GCC and Intel OpenMP backends is a bad idea.

#]=======================================================================]

cmake_minimum_required(VERSION 3.10)

# Modules
#
include(FindPackageHandleStandardArgs)

# Functions
#
function(__mkl_find_library _name)
    find_library(${_name}
        NAMES ${ARGN}
        HINTS ${MKL_ROOT}
        PATH_SUFFIXES ${_mkl_libpath_suffix}
                      lib
        )
    mark_as_advanced(${_name})
endfunction()

# External Find packages
#
find_package(Threads)
find_package(MPI COMPONENTS CXX)
find_package(OpenMP COMPONENTS CXX)

# Options
#
# The `NOT DEFINED` guards on CACHED variables are needed to make sure that 
# normal variables of the same name always take precedence*.
#
# * There are many caveats with CACHE variables in CMake. Before version 
#   3.12, both `option()` and `set(... CACHE ...)` would override normal 
#   variables if cached equivalents don't exist or they exisit but their type 
#   is not specified (e.g. command line arguments: -DFOO=ON instead of 
#   -DFOO:BOOL=ON). For 3.13 with policy CMP0077, `option()` no longer overrides 
#   normal variables of the same name. `set(... CACHE ...)` is still stuck with 
#   the old behaviour. 
#
#   https://cmake.org/cmake/help/v3.15/command/set.html#set-cache-entry
#   https://cmake.org/cmake/help/v3.15/policy/CMP0077.html
#
if(NOT DEFINED MKL_ROOT)
    set(MKL_ROOT $ENV{MKLROOT} CACHE PATH "MKL's root directory.")
endif()

# Determine MKL's library folder
#
set(_mkl_libpath_suffix "lib/intel64")
if(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32 bit
    set(_mkl_libpath_suffix "lib/ia32")
endif()

if(WIN32)
    string(APPEND _mkl_libpath_suffix "_win")
elseif(APPLE)
    string(APPEND _mkl_libpath_suffix "_mac")
else()
    string(APPEND _mkl_libpath_suffix "_lin")
endif()

# Find MKL header
#
find_path(MKL_INCLUDE_DIR mkl.h
    HINTS ${MKL_ROOT}/include
    )
mark_as_advanced(MKL_INCLUDE_DIR)

# BLAS components (core MKL)
#
__mkl_find_library(MKL_CORE_LIB mkl_core)

__mkl_find_library(MKL_INTERFACE_32BIT_LIB mkl_intel_lp64)
__mkl_find_library(MKL_INTERFACE_64BIT_LIB mkl_intel_ilp64)

__mkl_find_library(MKL_SEQ_LIB mkl_sequential)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    __mkl_find_library(MKL_OMP_LIB mkl_gnu_thread)
else()
    __mkl_find_library(MKL_OMP_LIB mkl_intel_thread)
endif()

# BLACS components
#
execute_process(COMMAND mpirun --version OUTPUT_VARIABLE MPIRUN_OUTPUT)
string(FIND "${MPIRUN_OUTPUT}" "Open MPI" _ompi_pos)
if(_ompi_pos STREQUAL "-1")  # MPICH
    if(APPLE)
        __mkl_find_library(MKL_BLACS_32BIT_LIB mkl_blacs_mpich_lp64)
        __mkl_find_library(MKL_BLACS_64BIT_LIB mkl_blacs_mpich_ilp64)
    else()
        __mkl_find_library(MKL_BLACS_32BIT_LIB mkl_blacs_intelmpi_lp64)
        __mkl_find_library(MKL_BLACS_64BIT_LIB mkl_blacs_intelmpi_ilp64)
    endif()
else()                      # OpenMPI
    if(APPLE)
        message(FATAL_ERROR "Only MPICH is supported on Apple.")
    endif()
     __mkl_find_library(MKL_BLACS_32BIT_LIB mkl_blacs_openmpi_lp64)
     __mkl_find_library(MKL_BLACS_64BIT_LIB mkl_blacs_openmpi_ilp64)
endif()

# ScaLAPACK components
#
__mkl_find_library(MKL_SCALAPACK_32BIT_LIB mkl_scalapack_lp64)
__mkl_find_library(MKL_SCALAPACK_64BIT_LIB mkl_scalapack_ilp64)

# Determine target existence for all components and define them if they do
#
foreach(_bits "32BIT" "64BIT")
    set(_mkl_interface_lib ${MKL_INTERFACE_${_bits}_LIB})
    set(_mkl_blacs_lib ${MKL_BLACS_${_bits}_LIB})
    set(_mkl_scalapack_lib ${MKL_SCALAPACK_${_bits}_LIB})

    foreach(_threading "SEQ" "OMP")
        set(_mkl_threading_lib ${MKL_${_threading}_LIB})

        set(_config "${_bits}_${_threading}")
        string(TOLOWER ${_config} _tgt_config)

        set(_omp_found_var "")
        set(_omp_target "")
        if(${_threading} STREQUAL "OMP")
            set(_omp_found_var "OpenMP_CXX_FOUND")
            set(_omp_target "OpenMP::OpenMP_CXX")
        endif()

        find_package_handle_standard_args(MKL_BLAS_${_config} REQUIRED_VARS _mkl_threading_lib 
                                                                            _mkl_interface_lib
                                                                            ${_omp_found_var})
        
        find_package_handle_standard_args(MKL_BLACS_${_config} REQUIRED_VARS _mkl_blacs_lib
                                                                             MKL_BLAS_${_config}_FOUND
                                                                             MPI_FOUND)

        find_package_handle_standard_args(MKL_SCALAPACK_${_config} REQUIRED_VARS _mkl_scalapack_lib
                                                                                 MKL_BLACS_${_config}_FOUND)
   
        if(MKL_BLAS_${_config}_FOUND AND NOT TARGET mkl::blas_${_tgt_config})
            add_library(mkl::blas_${_tgt_config} INTERFACE IMPORTED)
            set_target_properties(mkl::blas_${_tgt_config} PROPERTIES 
                INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${_mkl_interface_lib};${_mkl_threading_lib};${MKL_CORE_LIB};${_omp_target};Threads::Threads")
        endif()

        if(MKL_BLACS_${_config}_FOUND AND NOT TARGET mkl::blacs_${_tgt_config})
            add_library(mkl::blacs_${_tgt_config} INTERFACE IMPORTED)
            set_target_properties(mkl::blacs_${_tgt_config} PROPERTIES 
                INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${_mkl_interface_lib};${_mkl_threading_lib};${MKL_CORE_LIB};${_mkl_blacs_lib};${_omp_target};Threads::Threads")
        endif()

        if(MKL_SCALAPACK_${_config}_FOUND AND NOT TARGET mkl::scalapack_${_tgt_config})
            add_library(mkl::scalapack_${_tgt_config} INTERFACE IMPORTED)
            set_target_properties(mkl::scalapack_${_tgt_config} PROPERTIES 
                INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${_mkl_scalapack_lib};mkl::blacs_${_tgt_config}")
        endif()
    endforeach()
endforeach()

# Check if core libs were found
#
find_package_handle_standard_args(MKL REQUIRED_VARS MKL_INCLUDE_DIR
                                                    MKL_CORE_LIB
                                                    Threads_FOUND
                                      HANDLE_COMPONENTS)
