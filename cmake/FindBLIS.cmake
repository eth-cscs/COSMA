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
# FindBLIS
# -----------
#
# This module tries to find the BLIS library.
#
# The following variables are set
#
# ::
#
#   BLIS_FOUND           - True if blis is found
#   BLIS_LIBRARIES       - The required libraries
#   BLIS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   BLIS::blis

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
# if(NOT POLICY CMP0074)
set(_BLIS_PATHS ${BLIS_ROOT} 
                $ENV{BLIS_ROOT} 
                $ENV{BLISROOT}
                $ENV{BLIS_DIR}
                $ENV{BLISDIR})
# endif()

find_library(
    COSMA_BLIS_LINK_LIBRARIES
    NAMES "blis"
    HINTS ${_BLIS_PATHS}
    PATH_SUFFIXES "lib" "lib64" "blis/lib" "blis/lib64" "blis"
)
find_path(
    COSMA_BLIS_INCLUDE_DIRS
    NAMES "blis.h"
    HINTS ${_BLIS_PATHS}
    PATH_SUFFIXES "include" "blis" "blis/include" "include/blis"
)
find_path(
    COSMA_BLIS_CBLAS_INCLUDE_DIRS
    NAMES "cblas_blis.h" "cblas-blis.h" "cblas.h" 
    HINTS ${_BLIS_PATHS}
    PATH_SUFFIXES "include" "blis" "blis/include" "include/blis"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLIS REQUIRED_VARS COSMA_BLIS_INCLUDE_DIRS COSMA_BLIS_LINK_LIBRARIES COSMA_BLIS_CBLAS_INCLUDE_DIRS)

# add target to link against
if(NOT TARGET BLIS::blis)
  add_library(cosma::BLAS::BLIS::blis INTERFACE IMPORTED)
  add_library(cosma::BLAS::BLIS::blas ALIAS cosma::BLAS::BLIS::blis)
endif()
set_property(TARGET cosma::BLAS::BLIS::blis PROPERTY INTERFACE_LINK_LIBRARIES ${COSMA_BLIS_LINK_LIBRARIES})
set_property(TARGET cosma::BLAS::BLIS::blis PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${COSMA_BLIS_INCLUDE_DIRS} ${COSMA_BLIS_CBLAS_INCLUDE_DIRS})

# prevent clutter in cache
MARK_AS_ADVANCED(BLIS_FOUND COSMA_BLIS_LINK_LIBRARIES COSMA_BLIS_INCLUDE_DIRS COSMA_BLIS_CBLAS_INCLUDE_DIRS)
