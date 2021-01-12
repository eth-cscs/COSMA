# Set default to Release if none was specified and update the docs.
#
set(default_build_type ${CMAKE_BUILD_TYPE})
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(default_build_type "Release")
endif()
set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Profile." FORCE)

# Define a custom build type
#
#set( CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "" FORCE)
#set( CMAKE_C_FLAGS_PROFILE "${CMAKE_C_FLAGS_RELEASE}" CACHE STRING "" FORCE )
#set( CMAKE_EXE_LINKER_FLAGS_PROFILE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "" FORCE )
#set( CMAKE_SHARED_LINKER_FLAGS_PROFILE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "" FORCE )
#mark_as_advanced(CMAKE_CXX_FLAGS_PROFILE
#                 CMAKE_C_FLAGS_PROFILE
#                 CMAKE_EXE_LINKER_FLAGS_PROFILE
#                 CMAKE_SHARED_LINKER_FLAGS_PROFILE )
#
# use with $<$<CONFIG:Profile>:semiprof>
