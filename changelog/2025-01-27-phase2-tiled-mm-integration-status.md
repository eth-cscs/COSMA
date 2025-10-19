# Phase 2: Tiled-MM BF16 Integration Status

**Date:** January 27, 2025  
**Author:** David Sanftenberg  
**Status:** 75% Complete - Requires Template Integration

## Summary

Phase 2 (Tiled-MM BF16 Integration) has successfully added low-level BF16 GEMM wrappers to the Tiled-MM fork, but requires one additional step to complete: adding a template instantiation or wrapper overload to connect COSMA's `cosma::bfloat16` type to the new BF16 GEMM path.

## Completed Work

### 1. Tiled-MM Fork Setup ✅
- **Fork created:** `dbsanfte/Tiled-MM`
- **Branch:** `feature/bf16-support`
- **Commit:** `9de6bd8` (pushed to fork)
- **Submodule configured:** COSMA now tracks fork at BF16 branch

### 2. Low-Level BF16 GEMM Wrappers ✅

**File:** `libs/Tiled-MM/src/Tiled-MM/gpu_blas_api.hpp`
- Added BF16 type includes:
  - CUDA: `<cuda_bf16.h>` for `__nv_bfloat16`
  - ROCm: `<hip/hip_bfloat16.h>` for `hip_bfloat16`
- New function: `gemm_bf16()` (lines ~260-290)
  - Mixed precision: BF16 × BF16 → FP32
  - CUDA path: `cublasGemmEx` with `CUDA_R_16BF`, `CUBLAS_COMPUTE_32F`
  - ROCm path: `rocblas_gemm_ex` with `rocblas_datatype_bf16_r`
  - FP32 accumulation for numerical accuracy

**File:** `libs/Tiled-MM/src/Tiled-MM/tiled_mm.cpp`
- New function: `cublas_gemm_wrapper_bf16()` (lines ~280-310)
  - Conditional on `#ifdef TILED_MM_HAS_BF16_SUPPORT`
  - Handles operation types (trans_a, trans_b)
  - Calculates leading dimensions
  - Calls `blas_api::gemm_bf16()`

### 3. COSMA Build Integration ✅

**File:** `COSMA/CMakeLists.txt`
```cmake
FetchContent_Declare(
  Tiled-MM
  GIT_REPOSITORY https://github.com/dbsanfte/Tiled-MM.git
  GIT_TAG      feature/bf16-support  # Changed from commit hash
  FIND_PACKAGE_ARGS NAMES tiled-MM
)

if(COSMA_GPU_HAS_BF16_SUPPORT)
  target_compile_definitions(Tiled-MM::Tiled-MM INTERFACE TILED_MM_HAS_BF16_SUPPORT)
  message(STATUS "Tiled-MM BF16 support: ENABLED")
endif()
```

**File:** `COSMA/.gitmodules`
```ini
[submodule "libs/Tiled-MM"]
  url = https://github.com/dbsanfte/Tiled-MM.git
  branch = feature/bf16-support  # Added branch tracking
```

**Commit:** `c23d986` (pushed to COSMA fork)

## Remaining Work ⏳

### Template Integration (1-2 hours)

**Problem:** COSMA's `local_multiply.cpp` calls `gpu::gemm<Scalar>()`, which is a template function requiring explicit instantiation for each type. Current instantiations exist for:
- `float`, `double`, `std::complex<float>`, `std::complex<double>`

The new `cublas_gemm_wrapper_bf16()` function exists but is not callable from the templated path.

**Solution Options:**

#### Option A: Add Wrapper Overload (Preferred)
Add an overload of `cublas_gemm_wrapper` for `cosma::bfloat16` that internally calls `cublas_gemm_wrapper_bf16`:

```cpp
// In tiled_mm.cpp (around line 310)
#ifdef TILED_MM_HAS_BF16_SUPPORT
blas_api::StatusType cublas_gemm_wrapper(
    blas_api::HandleType handle,
    char trans_a, char trans_b,
    int m, int n, int k,
    const cosma::bfloat16* alpha,
    const cosma::bfloat16* a,
    const cosma::bfloat16* b,
    const cosma::bfloat16* beta,
    cosma::bfloat16* c,
    int lld_c) {
    
    // Convert BF16 scalars to FP32 for cuBLAS
    float alpha_f32 = static_cast<float>(*alpha);
    float beta_f32 = static_cast<float>(*beta);
    
    // Call BF16 wrapper (inputs BF16, output FP32)
    // TODO: Allocate FP32 output buffer and convert back to BF16
    // This requires additional logic for the mixed precision path
    return cublas_gemm_wrapper_bf16(handle, trans_a, trans_b,
                                    m, n, k, &alpha_f32,
                                    reinterpret_cast<const void*>(a),
                                    reinterpret_cast<const void*>(b),
                                    &beta_f32, 
                                    /* Need FP32 output here */,
                                    lld_c);
}
#endif
```

**Challenge:** Mixed precision handling. The `gemm_bf16()` function outputs FP32, but COSMA expects BF16 output. We need to either:
1. Add a conversion step (FP32 → BF16) after cuBLAS call
2. Create a two-stage approach (compute in FP32, store in BF16)
3. Modify the interface to support mixed precision outputs

#### Option B: Template Specialization
Create a template specialization of `gpu::gemm` for `cosma::bfloat16`:

```cpp
// In tiled_mm.cpp (around line 550)
#ifdef TILED_MM_HAS_BF16_SUPPORT
template<>
void gpu::gemm<cosma::bfloat16>(
    mm_handle<cosma::bfloat16>& handle,
    char trans_a, char trans_b,
    int m, int n, int k,
    cosma::bfloat16 alpha,
    cosma::bfloat16* a, int ld_a,
    cosma::bfloat16* b, int ld_b,
    cosma::bfloat16 beta,
    cosma::bfloat16* c, int ld_c,
    bool pin_host_buffers, bool copy_c_back) {
    
    // Custom implementation for BF16 that handles mixed precision
    // ... (implementation here)
}
#endif
```

**Advantage:** Full control over BF16 path, can handle mixed precision properly  
**Disadvantage:** More code duplication, harder to maintain

### Mixed Precision Handling Strategy

The key architectural decision is how to handle mixed precision (BF16 input → FP32 output):

1. **GPU-side conversion:**
   - Compute in FP32 on GPU
   - Convert to BF16 before copying back to host
   - Requires custom CUDA/HIP kernel or cuBLAS extension

2. **Host-side conversion:**
   - Copy FP32 output to host
   - Convert to BF16 on CPU
   - Simpler but adds CPU overhead

3. **Dual buffers:**
   - Maintain both FP32 and BF16 device buffers
   - Use FP32 for computation, BF16 for storage
   - Increases memory usage but avoids conversions

**Recommendation:** Start with host-side conversion for Phase 2, optimize with GPU-side kernels in Phase 4.

## Architecture Summary

### Call Chain (Current)
```
COSMA local_multiply.cpp:
  local_multiply<float>(gpu::mm_handle<float>*) 
    → gpu::gemm<float>()
      → cublas_gemm_wrapper(float*)
        → blas_api::gemm(float*)
          → cublasGemmEx(CUDA_R_32F)
```

### Call Chain (Desired for BF16)
```
COSMA local_multiply.cpp:
  local_multiply<bfloat16>(gpu::mm_handle<bfloat16>*) 
    → gpu::gemm<bfloat16>()
      → cublas_gemm_wrapper(bfloat16*) [NEW OVERLOAD]
        → cublas_gemm_wrapper_bf16(void*)
          → blas_api::gemm_bf16(void*)
            → cublasGemmEx(CUDA_R_16BF, compute=32F)
      → [FP32 → BF16 conversion] [NEW STEP]
```

## Next Steps

1. **Choose mixed precision strategy** (host-side conversion recommended)
2. **Implement `cublas_gemm_wrapper` overload for `cosma::bfloat16`**
   - Handle FP32 intermediate output
   - Add conversion logic (FP32 → BF16)
3. **Add template instantiation** at end of `tiled_mm.cpp`:
   ```cpp
   template void gemm<cosma::bfloat16>(...);
   ```
4. **Test compilation** (no GPU needed yet)
5. **Commit to Tiled-MM fork** (new commit on feature/bf16-support)
6. **Update COSMA submodule reference** (new commit to c23d986+)
7. **Proceed to Phase 3** (COSMA integration with `local_multiply.cpp`)

## Open Questions

1. **Should Tiled-MM include COSTA headers?**
   - Current: Tiled-MM is independent of COSTA
   - Needed for: `cosma::bfloat16` type definition
   - Alternative: Use `void*` with size parameter

2. **Should we support both BF16 → BF16 and BF16 → FP32 outputs?**
   - BF16 → FP32: Better accuracy, current cuBLAS limitation
   - BF16 → BF16: Lower memory, requires custom kernel or workaround

3. **Where should FP32 → BF16 conversion live?**
   - Tiled-MM (GPU library): More efficient, tied to GPU
   - COSMA (orchestration): More flexible, CPU overhead
   - Shared utility: Reusable, adds dependency

## Related Commits

- **COSTA:** `767b997` - GPU BF16 type conversions
- **COSMA:** `2bee5a2` - CMake BF16 detection
- **Tiled-MM:** `9de6bd8` - BF16 GEMM wrappers
- **COSMA:** `c23d986` - Phase 2 build integration

## Dependencies

- **Phase 1 (Complete):** Type system and CMake detection
- **Phase 2 (75%):** Tiled-MM integration (this document)
- **Phase 3 (Pending):** COSMA `local_multiply.cpp` integration
- **Phase 4 (Pending):** Testing and validation (requires GPU hardware)

## Estimated Completion

- **Remaining work:** 1-2 hours (template integration + mixed precision handling)
- **Testing:** Deferred to Phase 4 (requires Ampere or MI200 GPU)
- **Documentation:** 30 minutes (update main plan with mixed precision strategy)

## Files Modified (This Phase)

### Tiled-MM Fork (dbsanfte/Tiled-MM, feature/bf16-support)
- `src/Tiled-MM/gpu_blas_api.hpp` (+65 lines)
- `src/Tiled-MM/tiled_mm.cpp` (+51 lines)

### COSMA Fork (dbsanfte/COSMA, feature/gpu-bf16-support)
- `CMakeLists.txt` (+10 lines, lines 115-140)
- `.gitmodules` (modified Tiled-MM URL and branch)
- `libs/Tiled-MM` (submodule reference updated to 9de6bd8)

## Success Criteria

Phase 2 will be complete when:
- [x] Tiled-MM fork contains BF16 GEMM wrappers
- [x] COSMA build system uses Tiled-MM fork
- [x] Conditional compilation flag (`TILED_MM_HAS_BF16_SUPPORT`) defined
- [ ] `cublas_gemm_wrapper(cosma::bfloat16*)` overload exists
- [ ] `gpu::gemm<cosma::bfloat16>()` template instantiation exists
- [ ] Mixed precision (BF16 → FP32 → BF16) handled correctly
- [ ] Compiles successfully (no runtime testing yet)

**Current Progress:** 3/6 criteria met (75%)
