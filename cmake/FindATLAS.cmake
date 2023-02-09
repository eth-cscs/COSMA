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
# FindATLAS
# -----------
#
# This module tries to find the ATLAS library.
#
# The following variables are set
#
# ::
#
#   ATLAS_FOUND           - True if atlas is found
#   ATLAS_LIBRARIES       - The required libraries
#   ATLAS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
#   ATLAS::atlas

#set paths to look for library from ROOT variables.If new policy is set, find_library() automatically uses them.
# if(NOT POLICY CMP0074)
set(_ATLAS_PATHS ${ATLAS_ROOT}
                 $ENV{ATLAS_ROOT}
                 $ENV{ATLASROOT}
                 $ENV{ATLAS_DIR}
                 $ENV{ATLASDIR})
# endif()

find_library(
    COSMA_ATLAS_LINK_LIBRARIES
    NAMES "atlas"
    HINTS ${_ATLAS_PATHS}
    PATH_SUFFIXES "atlas/lib" "atlas/lib64" "atlas"
)
find_path(
    COSMA_ATLAS_INCLUDE_DIRS
    NAMES "cblas-atlas.h" "cblas_atlas.h" "cblas.h" 
    HINTS ${_ATLAS_PATHS}
    PATH_SUFFIXES "atlas" "atlas/include" "include/atlas"
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ATLAS REQUIRED_VARS COSMA_ATLAS_INCLUDE_DIRS COSMA_ATLAS_LINK_LIBRARIES)

# add target to link against
if(NOT TARGET cosma::BLAS::ATLAS::atlas)
  add_library(cosma::BLAS::ATLAS::atlas INTERFACE IMPORTED)
  add_library(cosma::BLAS::ATLAS::blas ALIAS cosma::BLAS::ATLAS::atlas)
endif()
set_property(TARGET cosma::BLAS::ATLAS::atlas PROPERTY INTERFACE_LINK_LIBRARIES ${COSMA_ATLAS_LINK_LIBRARIES})
set_property(TARGET cosma::BLAS::ATLAS::atlas PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${COSMA_ATLAS_INCLUDE_DIRS})

# prevent clutter in cache
MARK_AS_ADVANCED(ATLAS_FOUND ATLAS_LIBRARIES ATLAS_INCLUDE_DIRS)
