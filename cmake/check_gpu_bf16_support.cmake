# Check if the GPU backend supports BFloat16 operations
# Sets COSMA_GPU_HAS_BF16_SUPPORT to ON if supported, OFF otherwise
#
# Requirements:
# - CUDA: Version 11.0+ with Ampere (SM 80+) or newer GPU
# - ROCm: Version 4.5+ with CDNA2 (gfx90a) or newer GPU
#
# @author David Sanftenberg
# @date 2025-10-19

function(check_gpu_bf16_support)
    set(COSMA_GPU_HAS_BF16_SUPPORT OFF PARENT_SCOPE)
    
    if(COSMA_GPU_BACKEND STREQUAL "CUDA")
        # Check CUDA version (requires 11.0+ for BF16 support)
        find_package(CUDAToolkit QUIET)
        
        if(CUDAToolkit_FOUND)
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
                # CUDA 11.0+ has BF16 support, but we also need Ampere (SM 80+)
                # Try to detect GPU compute capability
                
                # First check if user set CMAKE_CUDA_ARCHITECTURES explicitly
                if(DEFINED CMAKE_CUDA_ARCHITECTURES)
                    foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
                        # Extract numeric part (e.g., "80" from "80-real" or just "80")
                        string(REGEX REPLACE "([0-9]+).*" "\\1" arch_num "${arch}")
                        if(arch_num GREATER_EQUAL 80)
                            set(COSMA_GPU_HAS_BF16_SUPPORT ON PARENT_SCOPE)
                            message(STATUS "GPU BF16 support: ENABLED (CUDA ${CUDAToolkit_VERSION}, SM ${arch_num})")
                            return()
                        endif()
                    endforeach()
                endif()
                
                # If not set, try to detect automatically using nvidia-smi
                find_program(NVIDIA_SMI "nvidia-smi")
                if(NVIDIA_SMI)
                    execute_process(
                        COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                        OUTPUT_VARIABLE GPU_COMPUTE_CAP
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET
                    )
                    
                    if(GPU_COMPUTE_CAP)
                        # Extract first GPU's compute capability (e.g., "8.0" -> 80)
                        string(REPLACE "." "" GPU_CC_NUM "${GPU_COMPUTE_CAP}")
                        string(STRIP "${GPU_CC_NUM}" GPU_CC_NUM)
                        string(SUBSTRING "${GPU_CC_NUM}" 0 2 GPU_CC_MAJOR)
                        
                        if(GPU_CC_MAJOR GREATER_EQUAL 80)
                            set(COSMA_GPU_HAS_BF16_SUPPORT ON PARENT_SCOPE)
                            message(STATUS "GPU BF16 support: ENABLED (CUDA ${CUDAToolkit_VERSION}, detected SM ${GPU_COMPUTE_CAP})")
                            return()
                        else()
                            message(STATUS "GPU BF16 support: DISABLED (CUDA ${CUDAToolkit_VERSION}, detected SM ${GPU_COMPUTE_CAP} < 8.0)")
                            message(STATUS "  BF16 requires NVIDIA Ampere (SM 8.0+) or newer GPU")
                            return()
                        endif()
                    endif()
                endif()
                
                # Couldn't detect GPU, warn user and enable conservatively
                message(WARNING "GPU BF16 support: Could not detect GPU compute capability")
                message(WARNING "  Set CMAKE_CUDA_ARCHITECTURES=80 (or higher) if you have Ampere+ GPU")
                message(WARNING "  BF16 GPU operations will be DISABLED (falling back to CPU)")
                set(COSMA_GPU_HAS_BF16_SUPPORT OFF PARENT_SCOPE)
            else()
                message(STATUS "GPU BF16 support: DISABLED (CUDA ${CUDAToolkit_VERSION} < 11.0)")
                message(STATUS "  BF16 requires CUDA 11.0+ with Ampere GPU")
            endif()
        else()
            message(WARNING "GPU BF16 support: Could not detect CUDA version")
        endif()
        
    elseif(COSMA_GPU_BACKEND STREQUAL "ROCM")
        # Check ROCm version (requires 4.5+ for BF16 support)
        find_package(hip QUIET)
        
        if(hip_FOUND)
            # ROCm doesn't have a clean version variable, try to get it from rocm_version.h
            find_file(ROCM_VERSION_H
                NAMES rocm_version.h rocm-core/rocm_version.h
                PATHS /opt/rocm/include
                NO_DEFAULT_PATH
            )
            
            if(ROCM_VERSION_H)
                file(STRINGS ${ROCM_VERSION_H} ROCM_VERSION_MAJOR REGEX "^#define ROCM_VERSION_MAJOR")
                file(STRINGS ${ROCM_VERSION_H} ROCM_VERSION_MINOR REGEX "^#define ROCM_VERSION_MINOR")
                string(REGEX REPLACE "^#define ROCM_VERSION_MAJOR ([0-9]+)" "\\1" ROCM_VER_MAJOR "${ROCM_VERSION_MAJOR}")
                string(REGEX REPLACE "^#define ROCM_VERSION_MINOR ([0-9]+)" "\\1" ROCM_VER_MINOR "${ROCM_VERSION_MINOR}")
                
                set(ROCM_VERSION "${ROCM_VER_MAJOR}.${ROCM_VER_MINOR}")
                
                if(ROCM_VERSION VERSION_GREATER_EQUAL "4.5")
                    # ROCm 4.5+ has BF16 support, but we also need CDNA2 (gfx90a)
                    # Try to detect GPU architecture
                    
                    if(DEFINED CMAKE_HIP_ARCHITECTURES)
                        # Check if gfx90a (MI200 series) is in the list
                        if("gfx90a" IN_LIST CMAKE_HIP_ARCHITECTURES OR
                           "gfx90a:xnack-" IN_LIST CMAKE_HIP_ARCHITECTURES OR
                           "gfx90a:xnack+" IN_LIST CMAKE_HIP_ARCHITECTURES)
                            set(COSMA_GPU_HAS_BF16_SUPPORT ON PARENT_SCOPE)
                            message(STATUS "GPU BF16 support: ENABLED (ROCm ${ROCM_VERSION}, gfx90a)")
                            return()
                        else()
                            message(STATUS "GPU BF16 support: DISABLED (ROCm ${ROCM_VERSION}, no gfx90a in CMAKE_HIP_ARCHITECTURES)")
                            message(STATUS "  BF16 requires AMD MI200 series (gfx90a) or newer GPU")
                            return()
                        endif()
                    endif()
                    
                    # Try to detect automatically using rocminfo
                    find_program(ROCMINFO "rocminfo")
                    if(ROCMINFO)
                        execute_process(
                            COMMAND ${ROCMINFO}
                            OUTPUT_VARIABLE ROCMINFO_OUTPUT
                            ERROR_QUIET
                        )
                        
                        if(ROCMINFO_OUTPUT MATCHES "gfx90a")
                            set(COSMA_GPU_HAS_BF16_SUPPORT ON PARENT_SCOPE)
                            message(STATUS "GPU BF16 support: ENABLED (ROCm ${ROCM_VERSION}, detected gfx90a)")
                            return()
                        else()
                            message(STATUS "GPU BF16 support: DISABLED (ROCm ${ROCM_VERSION}, no gfx90a detected)")
                            message(STATUS "  BF16 requires AMD MI200 series (CDNA2) or newer GPU")
                            return()
                        endif()
                    endif()
                    
                    # Couldn't detect GPU, warn user
                    message(WARNING "GPU BF16 support: Could not detect GPU architecture")
                    message(WARNING "  Set CMAKE_HIP_ARCHITECTURES=gfx90a if you have MI200 series GPU")
                    message(WARNING "  BF16 GPU operations will be DISABLED (falling back to CPU)")
                    set(COSMA_GPU_HAS_BF16_SUPPORT OFF PARENT_SCOPE)
                else()
                    message(STATUS "GPU BF16 support: DISABLED (ROCm ${ROCM_VERSION} < 4.5)")
                    message(STATUS "  BF16 requires ROCm 4.5+ with MI200 series GPU")
                endif()
            else()
                message(WARNING "GPU BF16 support: Could not detect ROCm version")
            endif()
        else()
            message(WARNING "GPU BF16 support: Could not find HIP package")
        endif()
    else()
        # No GPU backend, BF16 GPU support not applicable
        message(STATUS "GPU BF16 support: N/A (no GPU backend selected)")
    endif()
endfunction()
