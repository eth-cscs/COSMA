# Fetch and Build OpenBLAS from Source with BF16 Support
#
# This module fetches OpenBLAS v0.3.28 or later (which includes BF16 support)
# and builds it from source with appropriate optimizations.
#
# Sets:
#   OPENBLAS_FOUND - TRUE if OpenBLAS was successfully built/found
#   OPENBLAS_HAS_BF16_SUPPORT - TRUE if OpenBLAS has BF16 API
#   OpenBLAS::OpenBLAS - Imported target for OpenBLAS

include(FetchContent)
include(CheckSymbolExists)

option(COSMA_BUILD_OPENBLAS_FROM_SOURCE "Build OpenBLAS from source for BF16 support" ON)
option(COSMA_OPENBLAS_USE_OPENMP "Build OpenBLAS with OpenMP threading" ON)

function(fetch_openblas_with_bf16)
    message(STATUS "Fetching OpenBLAS from source for BF16 support...")
    
    # OpenBLAS 0.3.27+ has sbgemm (BF16) support
    # Using v0.3.28 which is stable and has good BF16 support
    FetchContent_Declare(
        openblas
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG v0.3.28
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    
    # Configure OpenBLAS build options
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "" FORCE)
    
    # Threading model
    if(COSMA_OPENBLAS_USE_OPENMP)
        set(USE_OPENMP 1 CACHE STRING "" FORCE)
        set(USE_THREAD 1 CACHE STRING "" FORCE)
        message(STATUS "Building OpenBLAS with OpenMP threading")
    else()
        set(USE_OPENMP 0 CACHE STRING "" FORCE)
        set(USE_THREAD 0 CACHE STRING "" FORCE)
        message(STATUS "Building OpenBLAS without threading")
    endif()
    
    # Enable optimizations
    set(DYNAMIC_ARCH ON CACHE BOOL "" FORCE)  # Multi-architecture support
    set(TARGET "GENERIC" CACHE STRING "" FORCE)  # Auto-detect at runtime
    
    # Fetch and build
    FetchContent_MakeAvailable(openblas)
    
    # Check if OpenBLAS has BF16 support (sbgemm function)
    set(CMAKE_REQUIRED_INCLUDES ${openblas_SOURCE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES openblas)
    
    check_symbol_exists(cblas_sbgemm "cblas.h" OPENBLAS_HAS_SBGEMM)
    
    if(OPENBLAS_HAS_SBGEMM)
        set(OPENBLAS_HAS_BF16_SUPPORT TRUE PARENT_SCOPE)
        message(STATUS "OpenBLAS built with BF16 support (sbgemm)")
    else()
        set(OPENBLAS_HAS_BF16_SUPPORT FALSE PARENT_SCOPE)
        message(WARNING "OpenBLAS built WITHOUT BF16 support")
    endif()
    
    # Create imported target if not exists
    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS ALIAS openblas)
    endif()
    
    # Export variables
    set(OPENBLAS_FOUND TRUE PARENT_SCOPE)
    set(OPENBLAS_INCLUDE_DIR ${openblas_SOURCE_DIR} PARENT_SCOPE)
    set(OPENBLAS_LIBRARIES openblas PARENT_SCOPE)
    
    message(STATUS "OpenBLAS source build complete")
endfunction()

# Check if OpenBLAS is already available and has BF16 support
function(check_existing_openblas_bf16)
    find_package(OpenBLAS QUIET)
    
    if(OpenBLAS_FOUND OR OPENBLAS_FOUND)
        message(STATUS "Found existing OpenBLAS installation")
        
        # Try to detect BF16 support in existing installation
        set(CMAKE_REQUIRED_INCLUDES ${OPENBLAS_INCLUDE_DIR})
        set(CMAKE_REQUIRED_LIBRARIES ${OPENBLAS_LIBRARIES})
        
        # Check for cblas_sbgemm (BF16 GEMM function in OpenBLAS)
        check_symbol_exists(cblas_sbgemm "cblas.h" EXISTING_OPENBLAS_HAS_SBGEMM)
        
        if(EXISTING_OPENBLAS_HAS_SBGEMM)
            set(OPENBLAS_HAS_BF16_SUPPORT TRUE PARENT_SCOPE)
            message(STATUS "Existing OpenBLAS has BF16 support (sbgemm)")
            set(USE_EXISTING_OPENBLAS TRUE PARENT_SCOPE)
        else()
            message(STATUS "Existing OpenBLAS does NOT have BF16 support")
            set(USE_EXISTING_OPENBLAS FALSE PARENT_SCOPE)
        endif()
    else()
        set(USE_EXISTING_OPENBLAS FALSE PARENT_SCOPE)
    endif()
endfunction()

# Main logic
if(COSMA_BUILD_OPENBLAS_FROM_SOURCE)
    # Always build from source when requested
    fetch_openblas_with_bf16()
else()
    # Try to use existing OpenBLAS, fall back to building from source if needed
    check_existing_openblas_bf16()
    
    if(NOT USE_EXISTING_OPENBLAS OR NOT OPENBLAS_HAS_BF16_SUPPORT)
        message(STATUS "Building OpenBLAS from source (existing version lacks BF16 support)")
        fetch_openblas_with_bf16()
    endif()
endif()

# Export configuration for parent scope
set(OPENBLAS_FOUND ${OPENBLAS_FOUND} PARENT_SCOPE)
set(OPENBLAS_HAS_BF16_SUPPORT ${OPENBLAS_HAS_BF16_SUPPORT} PARENT_SCOPE)
set(OPENBLAS_INCLUDE_DIR ${OPENBLAS_INCLUDE_DIR} PARENT_SCOPE)
set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARIES} PARENT_SCOPE)
