# taken from https://github.com/pytorch/pytorch/blob/master/cmake/Modules/FindNCCL.cmake
# which is licensed under: https://github.com/pytorch/pytorch/blob/master/LICENSE

# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(COSMA_NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(COSMA_NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(COSMA_NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

if ($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()
list(APPEND COSMA_NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${COSMA_NCCL_ROOT})

find_path(COSMA_NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${COSMA_NCCL_INCLUDE_DIR})

if (COSMA_USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(COSMA_NCCL_LIBNAME "nccl_static")
  if (COSMA_NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(COSMA_NCCL_LIBNAME "nccl")
  if (COSMA_NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(COSMA_NCCL_LIBRARIES
  NAMES ${COSMA_NCCL_LIBNAME}
  HINTS ${COSMA_NCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG COSMA_NCCL_INCLUDE_DIRS COSMA_NCCL_LIBRARIES)

if(COSMA_NCCL_FOUND)  # obtaining NCCL version and some sanity checks
  set (COSMA_NCCL_HEADER_FILE "${COSMA_NCCL_INCLUDE_DIRS}/nccl.h")
  message (STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")
  set (OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list (APPEND CMAKE_REQUIRED_INCLUDES ${COSMA_NCCL_INCLUDE_DIRS})
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(NCCL_VERSION_CODE nccl.h NCCL_VERSION_DEFINED)

  if (COSMA_NCCL_VERSION_DEFINED)
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
    file(WRITE ${file} "
      #include <iostream>
      #include <nccl.h>
      int main()
      {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;

        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
")
    try_run(COSMA_NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
      RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
      CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${COSMA_NCCL_INCLUDE_DIRS}"
      LINK_LIBRARIES ${COSMA_NCCL_LIBRARIES})
    if (NOT COSMA_NCCL_VERSION_MATCHED)
      message(FATAL_ERROR "Found NCCL header version and library version do not match! \
(include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}) Please set NCCL_INCLUDE_DIR and NCCL_LIB_DIR manually.")
    endif()
    message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
  else()
    message(STATUS "NCCL version < 2.3.5-5")
  endif ()
  set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})

  if (NOT TARGET cosma::nccl)
    add_library(cosma::nccl INTERFACE IMPORTED)
  endif()
  set_property(cosma::nccl PROPERTY INTERFACE_LINK_LIBRARIES ${COSMA_NCCL_LIBRARIES})
  set_property(cosma::nccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${COSMA_NCCL_INCLUDE_DIRS})

  message(STATUS "Found NCCL (include: ${COSMA_NCCL_INCLUDE_DIRS}, library: ${COSMA_NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()
