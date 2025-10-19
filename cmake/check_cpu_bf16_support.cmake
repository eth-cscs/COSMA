# Check CPU BF16 Support (AVX512_BF16)
#
# This module detects whether the CPU supports native BF16 operations
# via the AVX512_BF16 instruction set extension.
#
# Sets:
#   COSMA_CPU_HAS_BF16 - TRUE if CPU supports AVX512_BF16
#   COSMA_CPU_BF16_FLAGS - Compiler flags to enable BF16 instructions

include(CheckCXXSourceRuns)

function(check_cpu_bf16_support)
    set(CMAKE_REQUIRED_FLAGS "-mavx512bf16")
    
    check_cxx_source_runs("
        #include <immintrin.h>
        #include <iostream>
        
        int main() {
            // Check for AVX512_BF16 support via CPUID
            unsigned int eax, ebx, ecx, edx;
            
            // Check if CPUID is available
            #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
                // EAX=7, ECX=1: Extended Features
                __asm__ __volatile__(
                    \"cpuid\"
                    : \"=a\"(eax), \"=b\"(ebx), \"=c\"(ecx), \"=d\"(edx)
                    : \"a\"(7), \"c\"(1)
                );
                
                // AVX512_BF16 is bit 5 of EAX
                bool has_avx512bf16 = (eax & (1 << 5)) != 0;
                
                if (has_avx512bf16) {
                    std::cout << \"CPU supports AVX512_BF16\" << std::endl;
                    return 0;
                } else {
                    std::cout << \"CPU does NOT support AVX512_BF16\" << std::endl;
                    return 1;
                }
            #else
                // Non-x86 architecture, no BF16 support
                std::cout << \"Non-x86 CPU, no BF16 support\" << std::endl;
                return 1;
            #endif
        }
    " COSMA_CPU_HAS_AVX512BF16_RUNTIME)
    
    if(COSMA_CPU_HAS_AVX512BF16_RUNTIME)
        set(COSMA_CPU_HAS_BF16 TRUE PARENT_SCOPE)
        set(COSMA_CPU_BF16_FLAGS "-mavx512bf16" PARENT_SCOPE)
        message(STATUS "CPU supports native BF16 (AVX512_BF16)")
    else()
        set(COSMA_CPU_HAS_BF16 FALSE PARENT_SCOPE)
        set(COSMA_CPU_BF16_FLAGS "" PARENT_SCOPE)
        message(STATUS "CPU does NOT support native BF16")
    endif()
endfunction()

# Alternative: Compile-time check (doesn't run code, only checks if intrinsics are available)
function(check_cpu_bf16_compile_support)
    set(CMAKE_REQUIRED_FLAGS "-mavx512bf16")
    
    check_cxx_source_compiles("
        #include <immintrin.h>
        
        int main() {
            // Test BF16 intrinsics compilation
            __m512bh a = _mm512_setzero_pbh();
            __m512bh b = _mm512_setzero_pbh();
            __m512 c = _mm512_dpbf16_ps(_mm512_setzero_ps(), a, b);
            return 0;
        }
    " COSMA_CPU_HAS_BF16_INTRINSICS)
    
    if(COSMA_CPU_HAS_BF16_INTRINSICS)
        set(COSMA_CPU_BF16_COMPILE_SUPPORT TRUE PARENT_SCOPE)
        message(STATUS "Compiler supports BF16 intrinsics (-mavx512bf16)")
    else()
        set(COSMA_CPU_BF16_COMPILE_SUPPORT FALSE PARENT_SCOPE)
        message(STATUS "Compiler does NOT support BF16 intrinsics")
    endif()
endfunction()
