# Sets LAPACK variables according to the given type.
# LAPACK type CARMA_LAPACK_TYPE can have the following values:
# - Compiler (Default): The compiler add the scalapack flag automatically
#                       therefore no extra link line has to be added.
# - MKL: Uses MKLROOT env. variable or MKL_ROOT variable to find MKL.
#        The type of threading for MKL has to be specified using the variable
#        MKL_THREADING (Values: Intel OpenMP, GNU OpenMP, Sequential)
# - openblas: Uses BLASROOT env. variable to find openblas.
# - Custom: A custom link line has to be specified through CARMA_SCALAPACK_LIB.
# CARMA_LAPACK_LIBRARY provides the generated link line for BLAS LAPACK.

include(utils)
include(CheckFunctionExists)

function(cosma_find_lapack)
  unset(CARMA_LAPACK_LIBRARY CACHE)
  setoption(CARMA_LAPACK_TYPE STRING "Compiler" "BLAS/LAPACK type setting")
  set_property(CACHE CARMA_LAPACK_TYPE PROPERTY STRINGS Compiler MKL Custom)

  if(CARMA_LAPACK_TYPE MATCHES "MKL")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set(MKL_THREADING_OPTIONS Sequential "Intel OpenMP")
      set(MKL_THREADING_DEFAULT "Intel OpenMP")
    else()
      set(MKL_THREADING_OPTIONS Sequential "GNU OpenMP" "Intel OpenMP")
      set(MKL_THREADING_DEFAULT "GNU OpenMP")
    endif()
    setoption(MKL_THREADING STRING "${MKL_THREADING_DEFAULT}" "MKL Threading support")
    set_property(CACHE MKL_THREADING PROPERTY STRINGS ${MKL_THREADING_OPTIONS})
    # find the MKL library
    # If the variable MKL_ROOT is not defined
    # it is defined with the value of the env. variable MKLROOT
    setoption(MKL_ROOT PATH $ENV{MKLROOT} "Intel MKL path")
    message("${MKL_ROOT}")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set(MKL_LIB_DIR "-L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib")
    else()
      set(MKL_LIB_DIR "-L${MKL_ROOT}/lib/intel64")
    endif()

    if(MKL_THREADING MATCHES "Sequential")
      set(MKL_THREAD_LIB "-lmkl_sequential")
    elseif(MKL_THREADING MATCHES "GNU OpenMP")
      set(MKL_THREAD_LIB "-lmkl_gnu_thread -fopenmp")
    elseif(MKL_THREADING MATCHES "Intel OpenMP")
      if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        setoption(INTEL_LIBS_ROOT PATH "/opt/intel/lib" "Path to Intel libraries")
        find_library(IOMP5_LIB iomp5 HINTS "${INTEL_LIBS_ROOT}" NO_DEFAULT_PATH)
        if (IOMP5_LIB MATCHES "IOMP5_LIB-NOTFOUND")
          message(FATAL_ERROR "libiomp5 not found, please set INTEL_LIBS_ROOT correctly")
        endif()
        set(IOMP5_LIB_INTERNAL "-Wl,-rpath,${INTEL_LIBS_ROOT} ${IOMP5_LIB}")
      else()
        set(IOMP5_LIB_INTERNAL "-liomp5")
      endif()
      set(MKL_THREAD_LIB "-lmkl_intel_thread ${IOMP5_LIB_INTERNAL}")
    endif()
    set(CARMA_LAPACK_INTERNAL "${MKL_LIB_DIR} -lmkl_intel_lp64 ${MKL_THREAD_LIB} -lmkl_core -lpthread -lm -ldl")
  elseif(CARMA_LAPACK_TYPE MATCHES "openblas")
      setoption(OPENBLAS_ROOT PATH $ENV{BLASROOT} "OpenBlas path")
      message("${OPENBLAS_ROOT}")
      include_directories(${OPENBLAS_ROOT}/include)
      set(CARMA_LAPACK_INTERNAL "-L${OPENBLAS_ROOT}/lib -llapack -lopenblas -lpthread")
  elseif(CARMA_LAPACK_TYPE STREQUAL "Custom")
    setoption(CARMA_BLAS_LAPACK_LIB STRING "" "BLAS and LAPACK link line for CARMA_LAPACK_TYPE = Custom")
    set(CARMA_LAPACK_INTERNAL "${CARMA_BLAS_LAPACK_LIB}")
  elseif(CARMA_LAPACK_TYPE STREQUAL "Compiler")
    set(CARMA_LAPACK_INTERNAL "")
  else()
    message(FATAL_ERROR "Unknown LAPACK type: ${CARMA_LAPACK_TYPE)}")
  endif()
  set(CARMA_LAPACK_LIBRARY "${CARMA_LAPACK_INTERNAL}" CACHE PATH "BLAS/LAPACK link line (autogenerated)")

  set(CMAKE_REQUIRED_LIBRARIES "${CARMA_LAPACK_LIBRARY}")

  unset(CARMA_CHECK_BLAS CACHE)
  unset(CARMA_CHECK_LAPACK CACHE)
  # Check if BLAS and LAPACK work
  CHECK_FUNCTION_EXISTS(dgemm_ CARMA_CHECK_BLAS)
  if (NOT CARMA_CHECK_BLAS)
    message(FATAL_ERROR "Blas not found.")
  endif()

  CHECK_FUNCTION_EXISTS(dpotrf_ CARMA_CHECK_LAPACK)
  if (NOT CARMA_CHECK_LAPACK)
    message(FATAL_ERROR "Lapack not found.")
  endif()
  unset(CMAKE_REQUIRED_LIBRARIES)

endfunction()
