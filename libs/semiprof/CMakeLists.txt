cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
    set(MASTER_PROJECT ON)
endif()

option(SEMIPROF_WITH_EXAMPLES "Enable examples." ${MASTER_PROJECT})
option(SEMIPROF_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)


project(semiprof VERSION 1.0 LANGUAGES CXX)

# Prevents target conflicts if project is nested multiple times.
#
set(INSTALLED_TARGETS_LIST "")
if (NOT TARGET semiprof)
    add_library(semiprof "${semiprof_SOURCE_DIR}/src/profiler.cpp")
    target_include_directories(semiprof PUBLIC 
      $<BUILD_INTERFACE:${semiprof_SOURCE_DIR}/include>
     )
    target_compile_definitions(semiprof PUBLIC SEMIPROF) # Turn on profiler
    set(INSTALLED_TARGETS_LIST "semiprof")
endif()

if(SEMIPROF_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SEMIPROF_WITH_INSTALL AND INSTALLED_TARGETS_LIST)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(TARGETS ${INSTALLED_TARGETS_LIST}
            EXPORT semiprof_targets 
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    install(EXPORT semiprof_targets
            FILE semiprofConfig.cmake
            NAMESPACE semiprof::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/semiprof")

    install(FILES "${semiprof_SOURCE_DIR}/include/semiprof/semiprof.hpp"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    write_basic_package_version_file(
      "${semiprof_BINARY_DIR}/semiprofConfigVersion.cmake"
      VERSION ${semiprof_VERSION}
      COMPATIBILITY SameMajorVersion)

    install(FILES "${semiprof_BINARY_DIR}/semiprofConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/semiprof")
endif()

