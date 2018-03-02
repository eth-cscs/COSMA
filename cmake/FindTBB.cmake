# Locate Intel Threading Building Blocks include paths and libraries
# FindTBB.cmake can be found at https://code.google.com/p/findtbb/
# Written by Hannes Hofmann <hannes.hofmann _at_ informatik.uni-erlangen.de>
# Improvements by Gino van den Bergen <gino _at_ dtecta.com>,
#   Florian Uhlig <F.Uhlig _at_ gsi.de>,
#   Jiri Marsik <jiri.marsik89 _at_ gmail.com>
# The MIT License
# Copyright (c) 2011 Hannes Hofmann

# This module can use the following variables:
# TBB_INSTALL_DIR
# TBB_LIB_DIR

# This module defines
# TBB_FOUND
#TBB_INSTALL_DIR
#TBB_INCLUDE_DIR
#TBB_LIBRARY_DIR
#_tbb_deps 
#_tbb_deps_debug

# Architecture detection
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(TBB_ARCHITECTURE "intel64")
else ()
  set(TBB_ARCHITECTURE "ia32")
endif ()	

set(_TBB_ARCHITECTURE ${TBB_ARCHITECTURE})
set(_TBB_LIB_NAME "tbb")
set(_TBB_LIB_MALLOC_NAME "${_TBB_LIB_NAME}malloc")
set(_TBB_LIB_DEBUG_NAME "${_TBB_LIB_NAME}_debug")
set(_TBB_LIB_MALLOC_DEBUG_NAME "${_TBB_LIB_MALLOC_NAME}_debug")

if (WIN32)
  set(_TBB_DEFAULT_INSTALL_DIR "C:/Program Files/Intel/TBB"
    "C:/Program Files (x86)/Intel/TBB")
  if (MSVC10)
    set(_TBB_COMPILER "vc10")
  endif (MSVC10)
  if (MSVC11)
    set(_TBB_COMPILER "vc11")
  endif( MSVC11)
  if (MSVC12)
    set(_TBB_COMPILER "vc12")
  endif (MSVC12)
endif (WIN32)

if (UNIX)
  set(_TBB_DEFAULT_INSTALL_DIR "/opt/intel/tbb"
    "/usr" "/usr/local" "/usr/local/tbb" $ENV{TBB_ROOT})
endif ()

#Try to find path automatically
find_path(_TBB_INSTALL_DIR
  name "include/tbb/tbb.h"
  PATHS ${_TBB_DEFAULT_INSTALL_DIR} $ENV{TBB_INSTALL_DIR} ENV CPATH
    NO_DEFAULT_PATH )

# sanity check
if (NOT _TBB_INSTALL_DIR)
  if (TBB_FIND_REQUIRED)
    message(FATAL_ERROR "Could NOT find TBB library.")
  endif ()
else ()
  set (TBB_FOUND "YES")
  message("-- Found TBB: ${_TBB_INSTALL_DIR}")
  # Look for include directory and set ${TBB_INCLUDE_DIR}
  set(TBB_INCLUDE_DIR "${_TBB_INSTALL_DIR}/include" )
  # Look for libraries
  find_library(TBB_LIBRARY_DIR
    name ${_TBB_LIB_NAME}
    PATHS $ENV{TBB_LIB_DIR} $ENV{LD_LIBRARY_PATH})
  
  #release
  set(_tbb_deps ${_TBB_LIB_NAME} ${_TBB_LIB_MALLOC_NAME} )
  #debug
  set(_tbb_deps_debug	${_TBB_LIB_DEBUG_NAME} ${_TBB_LIB_MALLOC_DEBUG_NAME})

endif ()
