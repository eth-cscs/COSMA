#  Copyright (c) 2019 ETH Zurich
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


#.rst:
# FindHIPLIBS
# -----------
#
# This module searches for the fftw3 library.
#
# The following variables are set
#
# ::
#
#   HIPLIBS_FOUND           - True if hiplibs is found
#   HIPLIBS_LIBRARIES       - The required libraries
#   HIPLIBS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   HIPLIBS::hiplibs

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
#if(NOT POLICY CMP0074)
set(_HIPLIBS_PATHS ${HIPLIBS_ROOT} $ENV{HIPLIBS_ROOT})
#endif()

if(NOT _HIPLIBS_PATHS)
    set(_HIPLIBS_PATHS $ENV{ROCM_PATH} $ENV{ROCM_HOME} /opt/rocm)
endif()

if(NOT _RCCL_PATHS)
    set(_RCCL_PATHS $ENV{RCCL_HOME} $ENV{ROCM_PATH} $ENV{ROCM_HOME} /opt/rocm)
endif()

find_path(
    HIPLIBS_HIP_INCLUDE_DIRS
    NAMES "hip/hip_runtime_api.h"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/include" "include"
)
find_library(
    HIPLIBS_HIP_LIBRARY
    NAMES "amdhip64" "hip_hcc"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/lib" "lib" "lib64"
)
find_path(
    HIPLIBS_RCCL_INCLUDE_DIRS
    NAMES "rccl.h"
    HINTS ${_RCCL_PATHS}
    PATH_SUFFIXES "include"
)
find_library(
    HIPLIBS_RCCL_LIBRARY
    NAMES "rccl"
    HINTS ${_RCCL_PATHS}
    PATH_SUFFIXES "lib" "lib64"
)
find_library(
    HIPLIBS_ROCTX_LIBRARY
    NAMES "roctx64"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/lib" "lib" "lib64"
)
find_library(
    HIPLIBS_ROCTRACER_LIBRARY
    NAMES "roctracer64"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/lib" "lib" "lib64"
)
find_library(
    HIPLIBS_HSA_LIBRARY
    NAMES "hsa-runtime64"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hsa/lib" "lib" "lib64"
)
find_library(
    HIPLIBS_THUNK_LIBRARY
    NAMES "hsakmt"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hsa/lib" "lib" "lib64"
)
find_path(
    HIPLIBS_HSA_INCLUDE_DIRS
    NAMES "hsa/hsa.h"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/include" "include"
)
find_path(
    HIPLIBS_ROCTX_INCLUDE_DIRS
    NAMES "roctracer/roctx.h"
    HINTS ${_HIPLIBS_PATHS}
    PATH_SUFFIXES "hip/include" "include"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIPLIBS REQUIRED_VARS HIPLIBS_HIP_INCLUDE_DIRS HIPLIBS_HIP_LIBRARY HIPLIBS_ROCTX_LIBRARY HIPLIBS_ROCTRACER_LIBRARY HIPLIBS_RCCL_LIBRARY HIPLIBS_HSA_LIBRARY HIPLIBS_THUNK_LIBRARY HIPLIBS_HSA_INCLUDE_DIRS)


if(HIPLIBS_HIP_LIBRARY AND HIPLIBS_HSA_LIBRARY AND HIPLIBS_THUNK_LIBRARY AND HIPLIBS_ROCTX_LIBRARY AND HIPLIBS_ROCTRACER_LIBRARY)
    set(HIPLIBS_LIBRARIES ${HIPLIBS_HIP_LIBRARY} ${HIPLIBS_HSA_LIBRARY} ${HIPLIBS_THUNK_LIBRARY} ${HIPLIBS_ROCTX_LIBRARY} ${HIPLIBS_ROCTRACER_LIBRARY} ${HIPLIBS_RCCL_LIBRARY} CACHE STRING "Path to libraries.")
else()
    set(HIPLIBS_LIBRARIES HIPLIBS_LIBRARIES-NOTFOUND CACHE STRING "Path to libraries.")
endif()

if(HIPLIBS_HIP_INCLUDE_DIRS AND HIPLIBS_HSA_INCLUDE_DIRS AND HIPLIBS_ROCTX_INCLUDE_DIRS AND HIPLIBS_RCCL_INCLUDE_DIRS)
    set(HIPLIBS_INCLUDE_DIRS ${HIPLIBS_HIP_INCLUDE_DIRS} ${HIPLIBS_HSA_INCLUDE_DIRS} ${HIPLIBS_ROCTX_INCLUDE_DIRS} ${HIPLIBS_RCCL_INCLUDE_DIRS} CACHE STRING "Path to files.")
else()
    set(HIPLIBS_INCLUDE_DIRS HIPLIBS_INCLUDE_DIRS-NOTFOUND CACHE STRING "Path to files.")
endif()

# add target to link against
if(HIPLIBS_FOUND)
    if(NOT TARGET HIPLIBS::hiplibs)
        add_library(HIPLIBS::hiplibs INTERFACE IMPORTED)
    endif()
    set_property(TARGET HIPLIBS::hiplibs PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HIPLIBS_INCLUDE_DIRS})
    set_property(TARGET HIPLIBS::hiplibs PROPERTY INTERFACE_LINK_LIBRARIES ${HIPLIBS_LIBRARIES})
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(HIPLIBS_FOUND HIPLIBS_LIBRARIES HIPLIBS_INCLUDE_DIRS HIPLIBS_HIP_INCLUDE_DIRS HIPLIBS_HIP_LIBRARY HIPLIBS_HSA_LIBRARY HIPLIBS_THUNK_LIBRARY HIPLIBS_HSA_INCLUDE_DIRS HIPLIBS_RCCL_INCLUDE_DIRS HIPLIBS_ROCTX_INCLUDE_DIRS)
