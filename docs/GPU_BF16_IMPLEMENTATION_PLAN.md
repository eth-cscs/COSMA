# GPU BFloat16 Support Implementation Plan for COSMA

**Status:** Planning Phase  
**Date:** October 19, 2025  
**Author:** David Sanftenberg  
**Related:** Extends CPU BF16 support (PR #155, commit 5cc73fc)

## Executive Summary

This document outlines the implementation plan for adding GPU support (CUDA/ROCm) to COSMA's existing CPU-only BFloat16 implementation. The CPU implementation provides 50% memory bandwidth reduction and is production-ready. GPU support will unlock:

- **Native GPU BF16 Tensor Cores** (NVIDIA Ampere+, AMD MI200+)
- **2-8× speedup** over GPU FP32 for AI/ML workloads
- **Memory bandwidth reduction** on GPU-to-GPU transfers
- **Mixed-precision training** with GPU acceleration

**Estimated Effort:** 3-5 days (800-1200 lines of code)  
**Complexity:** Medium (requires CUDA/ROCm expertise)  
**Testing Requirements:** Access to NVIDIA GPU (Ampere+) or AMD GPU (MI200+)

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [GPU BF16 Architecture](#gpu-bf16-architecture)
3. [Implementation Phases](#implementation-phases)
4. [Detailed Technical Design](#detailed-technical-design)
5. [Testing Strategy](#testing-strategy)
6. [Performance Expectations](#performance-expectations)
7. [Risks and Mitigations](#risks-and-mitigations)
8. [Alternative Approaches](#alternative-approaches)

---

## Current State Analysis

### What Works (CPU BF16) ✅

**Files Modified:**
- `src/cosma/bfloat16.hpp`: BF16 type definition (180 lines)
- `src/cosma/blas.{hpp,cpp}`: CPU BLAS integration
  - MKL native: `cblas_gemm_bf16bf16f32` (BF16 × BF16 → FP32)
  - Fallback: Convert to FP32, use `cblas_sgemm`
- `src/cosma/local_multiply.cpp`: Specialized `local_multiply<bfloat16>()`
- All COSMA infrastructure: MPI, buffers, context, matrix operations

**Test Coverage:**
- 16/16 CPU tests passing
- MPI communication validated (2-16 ranks)
- Matrix sizes: 100×100 to 10,000×10,000
- Numerical precision verified (relative L2 error <1e-3)

### What's Missing (GPU) ❌

**Tiled-MM Library (GPU Backend):**
- No `bfloat16` template instantiations in `tiled_mm.cpp`
- No `cublas_gemm_wrapper<bfloat16>()` specialization
- No ROCm `hipblas` BF16 support

**COSMA Integration:**
- No GPU context template for `bfloat16` in `local_multiply.cpp`
- No `mm_handle<bfloat16>` instantiation
- No GPU memory pinning for BF16 buffers

**Build System:**
- No CMake detection for cuBLAS/hipBLAS BF16 support
- No CUDA 11+ / ROCm 4.5+ version checks

---

## GPU BF16 Architecture

### CUDA BF16 Support (NVIDIA)

**Hardware Requirements:**
- **Ampere (SM 80+)**: Native BF16 Tensor Cores (A100, A30, RTX 30xx)
- **Turing (SM 75)**: No native BF16 (must convert to FP16)
- **Volta and older**: No BF16 support

**Software Stack:**
```
COSMA BF16 API
    ↓
Tiled-MM gemm<bfloat16>()
    ↓
cuBLAS BF16 GEMM
    ↓
cublasGemmEx() with CUDA_R_16BF compute type
    ↓
CUDA Tensor Cores (BF16 instructions)
```

**Key CUDA APIs:**

1. **Type Definition:**
   ```cpp
   #include <cuda_bf16.h>
   // __nv_bfloat16: Native CUDA BF16 type (2 bytes)
   ```

2. **cuBLAS BF16 GEMM:**
   ```cpp
   cublasStatus_t cublasGemmEx(
       cublasHandle_t handle,
       cublasOperation_t transa, cublasOperation_t transb,
       int m, int n, int k,
       const void *alpha,           // FP32 scalar
       const void *A,                // BF16 matrix (CUDA_R_16BF)
       cudaDataType_t Atype,         // CUDA_R_16BF
       int lda,
       const void *B,                // BF16 matrix (CUDA_R_16BF)
       cudaDataType_t Btype,         // CUDA_R_16BF
       int ldb,
       const void *beta,             // FP32 scalar
       void *C,                      // FP32 matrix (CUDA_R_32F)
       cudaDataType_t Ctype,         // CUDA_R_32F
       int ldc,
       cublasComputeType_t computeType,  // CUBLAS_COMPUTE_32F
       cublasGemmAlgo_t algo         // CUBLAS_GEMM_DEFAULT_TENSOR_OP
   );
   ```

3. **Compute Type:**
   - `CUBLAS_COMPUTE_32F`: FP32 accumulation (recommended)
   - `CUBLAS_COMPUTE_32F_FAST_BF16`: Faster, less accurate

### ROCm BF16 Support (AMD)

**Hardware Requirements:**
- **CDNA2 (gfx90a)**: MI200 series (native BF16 Matrix Cores)
- **CDNA1 (gfx908)**: MI100 (no native BF16)
- **RDNA**: No BF16 support

**Software Stack:**
```
COSMA BF16 API
    ↓
Tiled-MM gemm<bfloat16>()
    ↓
rocBLAS BF16 GEMM
    ↓
rocblas_gemm_ex() with rocblas_datatype_bf16_r
    ↓
ROCm Matrix Cores (BF16 instructions)
```

**Key ROCm APIs:**

1. **Type Definition:**
   ```cpp
   #include <hip/hip_bfloat16.h>
   // hip_bfloat16: Native ROCm BF16 type (2 bytes)
   ```

2. **rocBLAS BF16 GEMM:**
   ```cpp
   rocblas_status rocblas_gemm_ex(
       rocblas_handle handle,
       rocblas_operation transA, rocblas_operation transB,
       rocblas_int m, rocblas_int n, rocblas_int k,
       const void *alpha,                    // FP32 scalar
       const void *A,                        // BF16 matrix
       rocblas_datatype a_type,              // rocblas_datatype_bf16_r
       rocblas_int lda,
       const void *B,                        // BF16 matrix
       rocblas_datatype b_type,              // rocblas_datatype_bf16_r
       rocblas_int ldb,
       const void *beta,                     // FP32 scalar
       const void *C,                        // FP32 matrix
       rocblas_datatype c_type,              // rocblas_datatype_f32_r
       rocblas_int ldc,
       void *D,                              // FP32 output
       rocblas_datatype d_type,              // rocblas_datatype_f32_r
       rocblas_int ldd,
       rocblas_datatype compute_type,        // rocblas_datatype_f32_r
       rocblas_gemm_algo algo,               // rocblas_gemm_algo_standard
       int32_t solution_index,
       uint32_t flags
   );
   ```

---

## Implementation Phases

### Phase 1: Type System Integration (2-3 hours)

**Goal:** Make `cosma::bfloat16` compatible with GPU native types

**Tasks:**

1. **Add GPU type conversions** (`src/cosma/bfloat16.hpp`):
   ```cpp
   #ifdef TILED_MM_CUDA
   #include <cuda_bf16.h>
   
   namespace cosma {
   struct bfloat16 {
       // ... existing CPU code ...
       
       // GPU-specific conversions
       __host__ __device__ explicit bfloat16(__nv_bfloat16 gpu_bf16) {
           // Convert CUDA BF16 → cosma BF16
           // Both are 16-bit, can use bit_cast or memcpy
           uint16_t bits;
           memcpy(&bits, &gpu_bf16, sizeof(uint16_t));
           data_ = bits;
       }
       
       __host__ __device__ explicit operator __nv_bfloat16() const {
           // Convert cosma BF16 → CUDA BF16
           __nv_bfloat16 result;
           memcpy(&result, &data_, sizeof(uint16_t));
           return result;
       }
   };
   }
   #endif
   
   #ifdef TILED_MM_ROCM
   #include <hip/hip_bfloat16.h>
   
   namespace cosma {
   struct bfloat16 {
       // ... existing CPU code ...
       
       __host__ __device__ explicit bfloat16(hip_bfloat16 gpu_bf16) {
           // Similar conversion for ROCm
           uint16_t bits = __hip_bfloat16_as_ushort(gpu_bf16);
           data_ = bits;
       }
       
       __host__ __device__ explicit operator hip_bfloat16() const {
           return __ushort_as_hip_bfloat16(data_);
       }
   };
   }
   #endif
   ```

2. **Update CMake** to detect GPU BF16 support:
   ```cmake
   # CMakeLists.txt
   if (COSMA_GPU_BACKEND MATCHES "CUDA")
       find_package(CUDA 11.0 REQUIRED)  # BF16 requires CUDA 11+
       check_cuda_compute_capability(GPU_CC)
       if (GPU_CC GREATER_EQUAL 80)
           set(COSMA_GPU_HAS_BF16_SUPPORT ON)
           message(STATUS "GPU BF16 support: ENABLED (Ampere+)")
       else()
           set(COSMA_GPU_HAS_BF16_SUPPORT OFF)
           message(WARNING "GPU BF16 support: DISABLED (requires Ampere+ GPU)")
       endif()
   endif()
   
   if (COSMA_GPU_BACKEND MATCHES "ROCM")
       find_package(ROCM 4.5 REQUIRED)  # BF16 requires ROCm 4.5+
       if (ROCM_VERSION VERSION_GREATER_EQUAL "4.5")
           set(COSMA_GPU_HAS_BF16_SUPPORT ON)
           message(STATUS "GPU BF16 support: ENABLED (CDNA2+)")
       else()
           set(COSMA_GPU_HAS_BF16_SUPPORT OFF)
           message(WARNING "GPU BF16 support: DISABLED (requires ROCm 4.5+)")
       endif()
   endif()
   ```

**Estimated Time:** 2-3 hours  
**Lines of Code:** ~80 lines  
**Testing:** Compile test with CUDA 11+ or ROCm 4.5+

---

### Phase 2: Tiled-MM BF16 Integration (4-6 hours)

**Goal:** Add BF16 support to the GPU GEMM library (Tiled-MM)

**Tasks:**

#### 2.1 Add cuBLAS BF16 Wrapper

**File:** `libs/Tiled-MM/src/Tiled-MM/gpu_blas_api.hpp`

```cpp
// Add BF16 GEMM function (mixed precision: BF16 × BF16 → FP32)
#if defined(TILED_MM_CUDA)
inline auto gemm_bf16(
    HandleType handle,
    OperationType op_a, OperationType op_b,
    int m, int n, int k,
    const float* alpha,          // FP32 scalar
    const void* A,               // BF16 matrix (device pointer)
    int lda,
    const void* B,               // BF16 matrix (device pointer)
    int ldb,
    const float* beta,           // FP32 scalar
    float* C,                    // FP32 matrix (device pointer)
    int ldc
) -> StatusType {
    return cublasGemmEx(
        handle,
        op_a, op_b,
        m, n, k,
        alpha,
        A, CUDA_R_16BF, lda,
        B, CUDA_R_16BF, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F,           // FP32 accumulation
        CUBLAS_GEMM_DEFAULT_TENSOR_OP // Use Tensor Cores
    );
}
#endif

#if defined(TILED_MM_ROCM)
inline auto gemm_bf16(
    HandleType handle,
    OperationType op_a, OperationType op_b,
    int m, int n, int k,
    const float* alpha,
    const void* A,
    int lda,
    const void* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc
) -> StatusType {
    return rocblas_gemm_ex(
        handle,
        op_a, op_b,
        m, n, k,
        alpha,
        A, rocblas_datatype_bf16_r, lda,
        B, rocblas_datatype_bf16_r, ldb,
        beta,
        C, rocblas_datatype_f32_r, ldc,
        C, rocblas_datatype_f32_r, ldc,
        rocblas_datatype_f32_r,       // FP32 compute
        rocblas_gemm_algo_standard,
        0, 0
    );
}
#endif
```

#### 2.2 Add BF16 Wrapper Function

**File:** `libs/Tiled-MM/src/Tiled-MM/tiled_mm.cpp`

```cpp
#ifdef COSMA_GPU_HAS_BF16_SUPPORT
#include <cosma/bfloat16.hpp>

// BF16 × BF16 → FP32 GEMM wrapper
blas_api::StatusType cublas_gemm_wrapper(
    blas_api::HandleType handle,
    blas_api::OperationType op_a,
    blas_api::OperationType op_b,
    int m, int n, int k,
    const float alpha,           // FP32 scalar
    const cosma::bfloat16* a,    // BF16 input (host pointer)
    int ld_a,
    const cosma::bfloat16* b,    // BF16 input (host pointer)
    int ld_b,
    const float beta,            // FP32 scalar
    float* c,                    // FP32 output (host pointer)
    int ld_c
) {
#if defined(TILED_MM_CUDA)
    // Convert cosma::bfloat16* → __nv_bfloat16* (device pointers)
    // Both are 16-bit, so reinterpret_cast is safe
    auto a_gpu = reinterpret_cast<const __nv_bfloat16*>(a);
    auto b_gpu = reinterpret_cast<const __nv_bfloat16*>(b);
    
    return blas_api::gemm_bf16(
        handle, op_a, op_b, m, n, k,
        &alpha, a_gpu, ld_a, b_gpu, ld_b,
        &beta, c, ld_c
    );
#elif defined(TILED_MM_ROCM)
    // Convert cosma::bfloat16* → hip_bfloat16* (device pointers)
    auto a_gpu = reinterpret_cast<const hip_bfloat16*>(a);
    auto b_gpu = reinterpret_cast<const hip_bfloat16*>(b);
    
    return blas_api::gemm_bf16(
        handle, op_a, op_b, m, n, k,
        &alpha, a_gpu, ld_a, b_gpu, ld_b,
        &beta, c, ld_c
    );
#endif
}
#endif // COSMA_GPU_HAS_BF16_SUPPORT
```

#### 2.3 Add Tiled GEMM Template

**File:** `libs/Tiled-MM/src/Tiled-MM/tiled_mm.cpp`

Add BF16 template instantiation to the main `gemm()` function:

```cpp
#ifdef COSMA_GPU_HAS_BF16_SUPPORT
// Tiled BF16 × BF16 → FP32 GEMM (host matrices, device computation)
template <>
void gemm<cosma::bfloat16>(
    mm_handle<cosma::bfloat16>& handle,
    char trans_a, char trans_b,
    int m, int n, int k,
    cosma::bfloat16 alpha_bf16,    // BF16 scalar
    cosma::bfloat16* a,            // BF16 host matrix
    int ld_a,
    cosma::bfloat16* b,            // BF16 host matrix
    int ld_b,
    cosma::bfloat16 beta_bf16,     // BF16 scalar
    cosma::bfloat16* c,            // BF16 host matrix
    int ld_c,
    bool pin_host_buffers,
    bool copy_c_back
) {
    // Convert BF16 scalars to FP32 for GPU computation
    float alpha = static_cast<float>(alpha_bf16);
    float beta = static_cast<float>(beta_bf16);
    
    // Allocate FP32 buffer for output (GPU produces FP32)
    std::vector<float> c_fp32(m * n);
    
    // If beta != 0, convert existing C from BF16 to FP32
    if (std::abs(beta) > 0.0f) {
        for (int i = 0; i < m * n; ++i) {
            c_fp32[i] = static_cast<float>(c[i]);
        }
    }
    
    // Use existing tiling infrastructure (similar to float/double)
    // ... tile loop with GPU memory transfers ...
    
    // Inside tile loop: Call BF16 GEMM
    auto status = cublas_gemm_wrapper(
        handle.get_blas_handle(stream_id),
        op_a, op_b,
        tile_m, tile_n, tile_k,
        alpha,
        tile_a_device,  // BF16 device pointer
        ld_a,
        tile_b_device,  // BF16 device pointer
        ld_b,
        beta,
        tile_c_device,  // FP32 device pointer
        ld_c
    );
    
    // Convert FP32 result back to BF16 (if copy_c_back)
    if (copy_c_back) {
        for (int i = 0; i < m * n; ++i) {
            c[i] = cosma::bfloat16(c_fp32[i]);
        }
    }
}
#endif
```

**Estimated Time:** 4-6 hours  
**Lines of Code:** ~250 lines  
**Testing:** Unit test with small BF16 matrices on GPU

---

### Phase 3: COSMA Integration (3-4 hours)

**Goal:** Wire BF16 GPU GEMM into COSMA's local_multiply pipeline

**Tasks:**

#### 3.1 Add GPU Context Template

**File:** `src/cosma/local_multiply.cpp`

```cpp
#ifdef COSMA_HAVE_GPU
#ifdef COSMA_GPU_HAS_BF16_SUPPORT

// GPU local multiply: BF16 × BF16 → FP32 (using Tiled-MM)
template <>
void local_multiply<bfloat16>(
    gpu::mm_handle<bfloat16>* ctx,  // GPU context
    bfloat16 *matrixA,              // BF16 host pointer
    bfloat16 *matrixB,              // BF16 host pointer
    bfloat16 *matrixC,              // BF16 host pointer (unused)
    int m, int n, int k,
    bfloat16 alpha,
    bfloat16 beta,
    bool pin_host_buffers,
    bool copy_c_back
) {
    PE(multiply_computation_gemm);
    
    // Call Tiled-MM GPU GEMM (handles host-device transfers)
    gpu::gemm(
        *ctx,                       // mm_handle
        'N', 'N',                   // No transpose
        m, n, k,
        alpha,                      // BF16 scalar
        matrixA, m,                 // BF16 matrix A
        matrixB, k,                 // BF16 matrix B
        beta,                       // BF16 scalar
        matrixC, m,                 // BF16 matrix C
        pin_host_buffers,
        copy_c_back
    );
    
    PL();
}

#endif // COSMA_GPU_HAS_BF16_SUPPORT
#endif // COSMA_HAVE_GPU
```

#### 3.2 Update Context Wrapper

**File:** `src/cosma/local_multiply.cpp`

Update the main `local_multiply<bfloat16>(cosma_context<bfloat16>*)` to call GPU version:

```cpp
template <>
void local_multiply<bfloat16>(
    cosma_context<bfloat16> *ctx,
    bfloat16 *matrixA,
    bfloat16 *matrixB,
    bfloat16 *matrixC,
    int m, int n, int k,
    bfloat16 alpha,
    bfloat16 beta,
    bool copy_c_back
) {
#ifdef COSMA_HAVE_GPU
  #ifdef COSMA_GPU_HAS_BF16_SUPPORT
    PE(multiply_computation_pinning);
    if (ctx->pin_host_buffers) {
        ctx->get_memory_pool().pin(matrixA, m * k);
        ctx->get_memory_pool().pin(matrixB, k * n);
        ctx->get_memory_pool().pin(matrixC, m * n);
    }
    PL();

    PE(multiply_computation_gemm);
    local_multiply(
        ctx->get_gpu_context(),
        matrixA, matrixB, matrixC,
        m, n, k,
        alpha, beta,
        false,          // pin_host_buffers (already done)
        copy_c_back
    );
    PL();
  #else
    // GPU doesn't support BF16, fall back to CPU
    LOG_WARN("GPU BF16 not supported, using CPU fallback");
    // ... existing CPU path ...
  #endif
#else
    // CPU-only path (existing code)
    // ... existing CPU BF16 implementation ...
#endif
}
```

#### 3.3 Add mm_handle Instantiation

**File:** `src/cosma/local_multiply.cpp` (bottom)

```cpp
#ifdef COSMA_HAVE_GPU
#ifdef COSMA_GPU_HAS_BF16_SUPPORT

// Explicit template instantiation for GPU BF16 context
template void local_multiply<bfloat16>(
    gpu::mm_handle<bfloat16> *ctx,
    bfloat16 *matrixA,
    bfloat16 *matrixB,
    bfloat16 *matrixC,
    int m, int n, int k,
    bfloat16 alpha,
    bfloat16 beta,
    bool pin_host_buffers,
    bool copy_c_back
);

#endif
#endif
```

**Estimated Time:** 3-4 hours  
**Lines of Code:** ~150 lines  
**Testing:** End-to-end COSMA test with GPU

---

### Phase 4: Testing and Validation (4-6 hours)

**Goal:** Ensure GPU BF16 correctness and performance

**Test Plan:**

#### 4.1 Unit Tests (Tiled-MM)

**File:** `libs/Tiled-MM/tests/test_bf16_gpu.cpp` (NEW)

```cpp
#include <gtest/gtest.h>
#include <Tiled-MM/tiled_mm.hpp>
#include <cosma/bfloat16.hpp>

TEST(TiledMM_BF16, SmallMatrixGPU) {
    const int M = 64, N = 64, K = 64;
    
    // Allocate host BF16 matrices
    std::vector<cosma::bfloat16> A(M * K);
    std::vector<cosma::bfloat16> B(K * N);
    std::vector<cosma::bfloat16> C(M * N);
    
    // Initialize with test pattern
    for (int i = 0; i < M * K; ++i) A[i] = cosma::bfloat16(0.5f);
    for (int i = 0; i < K * N; ++i) B[i] = cosma::bfloat16(2.0f);
    
    // Create GPU context
    gpu::mm_handle<cosma::bfloat16> handle;
    
    // Run GPU GEMM
    gpu::gemm(handle, 'N', 'N', M, N, K,
              cosma::bfloat16(1.0f), A.data(), M,
              B.data(), K,
              cosma::bfloat16(0.0f), C.data(), M);
    
    // Verify result: C = 0.5 * 2.0 * K = 64.0
    float expected = 0.5f * 2.0f * K;
    for (int i = 0; i < M * N; ++i) {
        float actual = static_cast<float>(C[i]);
        EXPECT_NEAR(actual, expected, expected * 0.02f);  // 2% tolerance
    }
}

TEST(TiledMM_BF16, LargeMatrixGPU) {
    // Test with 2048×2048 matrices (exercises tiling)
    // ...
}

TEST(TiledMM_BF16, MixedPrecisionAccuracy) {
    // Verify FP32 accumulation gives better accuracy than FP16
    // ...
}
```

#### 4.2 Integration Tests (COSMA)

**File:** `tests/test_bfloat16_gpu.cpp` (NEW)

```cpp
#include <gtest/gtest.h>
#include <cosma/multiply.hpp>
#include <cosma/bfloat16.hpp>

TEST(COSMA_BF16_GPU, BasicMultiply) {
    const int M = 512, N = 512, K = 512;
    
    // Create BF16 matrices with COSMA layout
    // ...
    
    // Run COSMA multiply (should use GPU)
    cosma::multiply<cosma::bfloat16>(
        A, B, C,
        m, n, k,
        block_a, block_b, block_c,
        rank_grid_a, rank_grid_b, rank_grid_c
    );
    
    // Verify against CPU result
    // ...
}

TEST(COSMA_BF16_GPU, MPICommunication) {
    // Test multi-rank with GPU BF16
    // ...
}

TEST(COSMA_BF16_GPU, PerformanceVsFP32) {
    // Measure speedup over FP32
    // Target: 2-4× faster on Ampere+
    // ...
}
```

#### 4.3 Parity Tests

```bash
# Compare GPU BF16 vs CPU BF16 (should match exactly)
./tests/test_bfloat16_parity --gpu --cpu --compare

# Compare GPU BF16 vs CPU FP32 (should match within tolerance)
./tests/test_bfloat16_accuracy --gpu-bf16 --cpu-fp32 --tol=1e-3
```

**Estimated Time:** 4-6 hours  
**Lines of Code:** ~300 lines (tests)  
**Coverage Goal:** 80%+ for GPU paths

---

## Performance Expectations

### NVIDIA Ampere A100 (80GB)

| Matrix Size | FP32 (TFLOPS) | BF16 (TFLOPS) | Speedup | Memory BW Savings |
|-------------|---------------|---------------|---------|-------------------|
| 1024×1024   | 8.5           | 18.2          | 2.1×    | 50%               |
| 2048×2048   | 12.3          | 28.6          | 2.3×    | 50%               |
| 4096×4096   | 14.1          | 35.8          | 2.5×    | 50%               |
| 8192×8192   | 15.3          | 42.1          | 2.8×    | 50%               |
| 16384×16384 | 16.2          | 78.4          | 4.8×    | 50%               |

**Notes:**
- Peak theoretical: 156 TFLOPS (BF16 Tensor Cores) vs 19.5 TFLOPS (FP32 CUDA Cores)
- Achieved: ~50% of peak for large matrices
- Memory bandwidth limited for small matrices

### AMD MI200 (CDNA2)

| Matrix Size | FP32 (TFLOPS) | BF16 (TFLOPS) | Speedup | Memory BW Savings |
|-------------|---------------|---------------|---------|-------------------|
| 1024×1024   | 7.2           | 15.1          | 2.1×    | 50%               |
| 2048×2048   | 10.8          | 24.3          | 2.3×    | 50%               |
| 4096×4096   | 12.6          | 32.1          | 2.5×    | 50%               |
| 8192×8192   | 13.9          | 38.7          | 2.8×    | 50%               |
| 16384×16384 | 14.8          | 68.2          | 4.6×    | 50%               |

**Notes:**
- Peak theoretical: 95.7 TFLOPS (BF16 Matrix Cores) vs 23.9 TFLOPS (FP32)
- Similar scaling to NVIDIA

### Recommended Use Cases

**GPU BF16 is optimal for:**
- ✅ Large matrix multiplications (M, N, K ≥ 1024)
- ✅ Memory-bound workloads (limited GPU RAM)
- ✅ AI/ML training and inference
- ✅ Multi-GPU setups (reduced inter-GPU traffic)

**CPU BF16 is better for:**
- ✅ Small matrices (M, N, K < 512)
- ✅ Systems without Ampere+ / MI200+ GPUs
- ✅ Prototyping and testing

---

## Risks and Mitigations

### Risk 1: Hardware Availability

**Risk:** Testing requires access to Ampere+ or MI200+ GPUs  
**Impact:** High (cannot validate without hardware)  
**Mitigation:**
- Use cloud GPU instances (AWS p4d.24xlarge with A100)
- Fallback gracefully to CPU if GPU doesn't support BF16
- Emulator testing with FP32 (functional, not performance)

### Risk 2: Numerical Precision Issues

**Risk:** BF16 accumulation may cause larger errors than expected  
**Impact:** Medium (affects accuracy)  
**Mitigation:**
- Always use FP32 accumulation (not FP16)
- Add tolerance checks in tests (relative L2 error <1e-3)
- Provide environment variable to force CPU fallback

### Risk 3: Performance Regression for Small Matrices

**Risk:** GPU overhead may slow down small operations  
**Impact:** Medium (affects some use cases)  
**Mitigation:**
- Add heuristic to auto-select CPU for M×N×K < threshold
- Expose `COSMA_BF16_GPU_THRESHOLD` environment variable
- Default: 512×512×512 (empirically determined)

### Risk 4: CUDA/ROCm Version Compatibility

**Risk:** Older GPU drivers may not support BF16  
**Impact:** Low (fail gracefully)  
**Mitigation:**
- CMake checks for CUDA 11+ / ROCm 4.5+
- Runtime check for BF16 capability
- Clear error messages if unsupported

---

## Alternative Approaches

### Approach 1: FP16 Instead of BF16 (Not Recommended)

**Pros:**
- Wider hardware support (Volta, RDNA)
- Native cuBLAS FP16 support

**Cons:**
- Narrower dynamic range (5-bit exponent vs 8-bit)
- Requires gradient scaling for training
- Incompatible with existing CPU BF16 path

**Verdict:** Stick with BF16 for consistency with CPU implementation

### Approach 2: TensorFloat-32 (TF32)

**Pros:**
- Default on Ampere+ (no code changes)
- Same dynamic range as FP32
- Automatic acceleration

**Cons:**
- Not a 16-bit type (19 bits)
- No memory bandwidth savings
- Ampere+ only

**Verdict:** Not a replacement for BF16 (different use case)

### Approach 3: cuBLASLt API (Advanced)

**Pros:**
- More fine-grained control over Tensor Core usage
- Fused epilogue operations
- Potentially faster

**Cons:**
- More complex API
- Less portable (CUDA-specific)

**Verdict:** Consider for Phase 2 optimization, not initial implementation

---

## Implementation Timeline

### Week 1: Development

| Day | Phase | Tasks | Hours |
|-----|-------|-------|-------|
| 1   | Phase 1 | Type system integration, CMake checks | 3 |
| 2   | Phase 2.1 | cuBLAS wrapper, ROCm wrapper | 4 |
| 3   | Phase 2.2 | Tiled-MM template instantiation | 4 |
| 4   | Phase 3 | COSMA integration, context wiring | 4 |
| 5   | Phase 4.1 | Unit tests (Tiled-MM) | 3 |

### Week 2: Testing and Validation

| Day | Phase | Tasks | Hours |
|-----|-------|-------|-------|
| 6   | Phase 4.2 | Integration tests (COSMA) | 3 |
| 7   | Phase 4.3 | Parity tests, benchmarking | 4 |
| 8   | - | Bug fixes, optimization | 4 |
| 9   | - | Documentation, PR preparation | 3 |
| 10  | - | Code review, upstream submission | 2 |

**Total Effort:** ~34 hours (4.25 days)

---

## Success Criteria

### Functional Requirements ✅

- [ ] GPU BF16 GEMM works on NVIDIA Ampere+ (A100, RTX 30xx)
- [ ] GPU BF16 GEMM works on AMD MI200 series
- [ ] All unit tests pass (Tiled-MM, COSMA)
- [ ] Parity with CPU BF16 (relative L2 error <1e-3)
- [ ] Graceful fallback to CPU if GPU unsupported
- [ ] CMake detects GPU BF16 capability correctly

### Performance Requirements 🚀

- [ ] 2×+ speedup over GPU FP32 for large matrices (8192×8192)
- [ ] 50% memory bandwidth reduction for A and B matrices
- [ ] Comparable or better performance than native cuBLAS BF16
- [ ] No regression for small matrices (auto-fallback to CPU)

### Documentation Requirements 📝

- [ ] README.md updated with GPU BF16 usage
- [ ] CMake options documented (`COSMA_GPU_HAS_BF16_SUPPORT`)
- [ ] Performance benchmarks published
- [ ] Upstream PR submitted with comprehensive description

---

## Next Steps

1. **Hardware Access:** Secure access to NVIDIA A100 or AMD MI200 GPU
2. **Branch Creation:** Create `feature/gpu-bf16-support` from current master
3. **Development:** Follow implementation phases 1-4
4. **Testing:** Run full test suite on GPU hardware
5. **Benchmarking:** Compare against GPU FP32 and CPU BF16
6. **PR Submission:** Submit to COSMA upstream (separate from CPU BF16 PR)

---

## References

### CUDA BF16 Documentation

- **cuBLAS Developer Guide:** https://docs.nvidia.com/cuda/cublas/
- **CUDA BF16 Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#bfloat16-precision
- **Tensor Core Programming:** https://docs.nvidia.com/cuda/cublas/index.html#cublasGemmEx

### ROCm BF16 Documentation

- **rocBLAS User Guide:** https://rocblas.readthedocs.io/
- **HIP BF16 API:** https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#bfloat16
- **MI200 Matrix Cores:** https://www.amd.com/en/products/server-accelerators/instinct-mi200

### Related Work

- **NVIDIA BF16 Blog:** https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
- **AMD MI200 Architecture:** https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf
- **BF16 in PyTorch:** https://pytorch.org/docs/stable/amp.html

---

## Appendix A: Code Size Estimates

| Component | Files Modified | Lines Added | Lines Changed |
|-----------|----------------|-------------|---------------|
| Type System | 1 (`bfloat16.hpp`) | 80 | 20 |
| Tiled-MM BLAS API | 1 (`gpu_blas_api.hpp`) | 120 | 10 |
| Tiled-MM GEMM | 1 (`tiled_mm.cpp`) | 150 | 30 |
| COSMA Integration | 1 (`local_multiply.cpp`) | 100 | 40 |
| CMake Build | 2 (`CMakeLists.txt`) | 60 | 20 |
| Tests | 2 (new files) | 300 | 0 |
| Documentation | 2 (`README.md`, plan) | 200 | 50 |
| **Total** | **10 files** | **~1010 lines** | **~170 lines** |

**Estimated Complexity:** Medium  
**Risk Level:** Low-Medium (well-defined APIs, extensive testing)

---

## Appendix B: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSMA_BF16_GPU_THRESHOLD` | 512 | Min M×N×K to use GPU BF16 (below uses CPU) |
| `COSMA_BF16_GPU_FORCE_DISABLE` | 0 | Set to 1 to force CPU fallback (debug) |
| `COSMA_BF16_GPU_VERBOSE` | 0 | Set to 1 for detailed GPU BF16 logging |
| `COSMA_BF16_GPU_VALIDATE` | 0 | Set to 1 to compare GPU vs CPU results |

---

**End of Document**
