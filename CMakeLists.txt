cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Check if COSMA is a subproject.
#
set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
  set(MASTER_PROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

# Options
#
option(COSMA_WITH_GPU "Use CUDA GPU backend." OFF)
option(COSMA_WITH_PROFILING "Use the semiprof profiler" OFF)
option(COSMA_WITH_TESTS "Generate the test target." ${MASTER_PROJECT})
option(COSMA_WITH_APPS "Generate the miniapp targets." ${MASTER_PROJECT})
option(COSMA_WITH_BENCHMARKS "Generate the benchmark targets." ${MASTER_PROJECT})
option(COSMA_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})
set(COSMA_WITH_MKL ON)

# we don't necessarily want to turn off MKL because maybe
# the user still wants to run pdgemm wrapper
# if(COSMA_WITH_GPU)
#     set(COSMA_WITH_MKL OFF)
# endif()


project(cosma VERSION 0.1 LANGUAGES CXX)


# Dependencies
#
find_package(MPI REQUIRED)

option(OPTIONS_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
add_subdirectory(libs/options)

option(GRID2GRID_WITH_INSTALL "" ${COSMA_WITH_INSTALL})
option(GRID2GRID_WITH_PROFILING "" ${COSMA_WITH_PROFILING})
add_subdirectory(libs/grid2grid)

if(COSMA_WITH_PROFILING)
    add_subdirectory(libs/semiprof)
endif()

if(COSMA_WITH_MKL)
    find_package(MKL REQUIRED)
    find_package(ScaLAPACK)  # optional
endif()

if(COSMA_WITH_GPU)
    option(TILEDMM_WITH_INSTALL "" ${COSMA_WITH_INSTALL}) 
    add_subdirectory(libs/Tiled-MM)
endif()


# COSMA
#
add_subdirectory(src/cosma)


if(COSMA_WITH_INSTALL)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(EXPORT cosma_targets
            FILE cosmaTargets.cmake
            NAMESPACE cosma::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cosma")

    install(DIRECTORY "${cosma_SOURCE_DIR}/src/cosma" 
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.hpp")

    write_basic_package_version_file(
        "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
        VERSION ${cosma_VERSION}
        COMPATIBILITY SameMajorVersion)

    configure_file("${cosma_SOURCE_DIR}/cmake/cosma.pc.in"
                   "${cosma_BINARY_DIR}/cosma.pc"
                   @ONLY)

    configure_file("${cosma_SOURCE_DIR}/cmake/cosmaConfig.cmake.in"
                   "${cosma_BINARY_DIR}/cosmaConfig.cmake"
                   @ONLY)

    install(FILES "${cosma_BINARY_DIR}/cosmaConfig.cmake"
                  "${cosma_BINARY_DIR}/cosmaConfigVersion.cmake"
                  "${cosma_SOURCE_DIR}/cmake/FindMKL.cmake"
                  "${cosma_SOURCE_DIR}/cmake/FindScaLAPACK.cmake"
                  "${cosma_SOURCE_DIR}/cmake/check_for_openmpi.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cosma")

    install(FILES "${cosma_BINARY_DIR}/cosma.pc"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif()

if(COSMA_WITH_TESTS)
    add_subdirectory(libs/gtest_mpi)
    enable_testing()
    add_subdirectory(tests)
endif()

if(COSMA_WITH_APPS)
    add_subdirectory(miniapp)
endif()

if(COSMA_WITH_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
