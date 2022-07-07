# Try to find RCCL
#
# The following variables are optionally searched for defaults
#  RCCL_ROOT_DIR: Base directory where all RCCL components are found
#  RCCL_INCLUDE_DIR: Directory where RCCL header is found
#  RCCL_LIB_DIR: Directory where RCCL library is found
#
# The following are set after configuration is done:
#  RCCL_FOUND
#  RCCL_INCLUDE_DIRS
#  RCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install RCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

if(DEFINED ENV{RCCL_ROOT_DIR})
    set(RCCL_ROOT_DIR $ENV{RCCL_ROOT_DIR})
endif()

if(DEFINED ENV{RCCL_INCLUDE_DIR})
    set(RCCL_INCLUDE_DIR $ENV{RCCL_INCLUDE_DIR})
endif()

if(DEFINED ENV{RCCL_LIB_DIR})
    set(RCCL_LIB_DIR $ENV{RCCL_LIB_DIR})
endif()

if(NOT DEFINED ENV{RCCL_ROOT_DIR} AND DEFINED ENV{ROCM_PATH})
    set(RCCL_ROOT_DIR $ENV{ROCM_PATH} CACHE PATH "Folder contains AMD RCCL")
endif()

if(NOT DEFINED ENV{RCCL_ROOT_DIR} AND NOT DEFINED ENV{ROCM_PATH})
    set(RCCL_ROOT_DIR "/opt/rocm")
endif()

find_path(RCCL_INCLUDE_DIRS
    NAMES rccl.h
    HINTS
    ${RCCL_INCLUDE_DIR}
    ${RCCL_ROOT_DIR}/include)

set(RCCL_LIBNAME "rccl")

if (DEFINED ENV{USE_STATIC_RCCL})
    message(STATUS "USE_STATIC_RCCL detected. Linking against static RCCL library")
    set(RCCL_LIBNAME "rccl_static")
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 ".a" )
else()
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 ".so" )
endif()

find_library(RCCL_LIBRARIES
    NAMES ${RCCL_LIBNAME}
    HINTS
    ${RCCL_LIB_DIR}
    ${RCCL_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL DEFAULT_MSG RCCL_INCLUDE_DIRS RCCL_LIBRARIES)

if (RCCL_FOUND)
    set(RCCL_HEADER_FILE "${RCCL_INCLUDE_DIR}/rccl.h")
    message(STATUS "Determining RCCL version from the header file: ${RCCL_HEADER_FILE}")
    file (STRINGS ${RCCL_HEADER_FILE} RCCL_MAJOR_VERSION_DEFINED
        REGEX "^[ \t]*#define[ \t]+RCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
    if (RCCL_MAJOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+RCCL_MAJOR[ \t]+" ""
            RCCL_MAJOR_VERSION ${RCCL_MAJOR_VERSION_DEFINED})
        message(STATUS "RCCL_MAJOR_VERSION: ${RCCL_MAJOR_VERSION}")
    endif()
    message(STATUS "Found RCCL (include: ${RCCL_INCLUDE_DIRS}, library: ${RCCL_LIBRARIES})")
    mark_as_advanced(RCCL_ROOT_DIR RCCL_INCLUDE_DIRS RCCL_LIBRARIES)
endif()
