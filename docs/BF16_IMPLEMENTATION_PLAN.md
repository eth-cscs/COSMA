# BF16 Matrix Multiplication Support in COSMA

## Objective
Add support for BF16 × BF16 → FP32 (with FP32 accumulation) matrix multiplication to COSMA.

## Background

### Current Type Support
COSMA currently supports:
- `float` (FP32)
- `double` (FP64)  
- `std::complex<float>`
- `std::complex<double>`

### Challenge: Mixed Precision
BF16 matmul with FP32 accumulation is a **mixed-precision operation**:
- **Inputs**: BF16 (16-bit bfloat16)
- **Output**: FP32 (32-bit float)
- **Accumulation**: FP32 (to avoid precision loss)

This differs from COSMA's current homogeneous type model where `Scalar` applies uniformly to A, B, C, alpha, and beta.

## Implementation Strategy

### Phase 1: BFloat16 Type Definition
**Goal**: Define and integrate bfloat16 type into COSMA's type system

**Files to modify**:
1. Create `src/cosma/bfloat16.hpp`:
   - Define `bfloat16` struct (16-bit: 1 sign, 8 exponent, 7 mantissa)
   - Conversion operators to/from float
   - Basic arithmetic operators (for compatibility)
   - OR: Use existing library (e.g., `__nv_bfloat16` from CUDA, or `bfloat16_t` from oneDNN)

2. `libs/COSTA/src/costa/grid2grid/mpi_type_wrapper.hpp`:
   ```cpp
   template <>
   struct mpi_type_wrapper<bfloat16> {
       static MPI_Datatype type() { 
           // BF16 is 16 bits, use MPI_UINT16_T or create custom type
           return MPI_UINT16_T;
       }
   };
   ```

**Decision Point**: 
- **Option A**: Use existing BF16 library (oneDNN, CUDA) - faster, tested
- **Option B**: Implement custom BF16 type - more control, no dependencies
- **Recommendation**: Start with Option B for CPU-only, add Option A for GPU later

### Phase 2: Mixed-Precision GEMM Interface
**Goal**: Add gemm variant that accepts BF16 inputs and produces FP32 outputs

**Challenge**: Current BLAS libraries (OpenBLAS, MKL, BLIS) have limited BF16 support:
- **MKL**: Has `cblas_gemm_bf16bf16f32` (BF16 × BF16 → FP32) since 2020+
- **OpenBLAS**: No native BF16 support (as of 0.3.x)
- **BLIS**: Experimental BF16 in some versions
- **oneDNN**: Full BF16 support via `dnnl_sgemm` with bf16 data types

**Files to modify**:

1. `src/cosma/blas.hpp`:
   ```cpp
   namespace cosma {
   
   // NEW: Mixed-precision BF16 × BF16 → FP32
   void gemm_bf16(const int M,
                  const int N,
                  const int K,
                  const float alpha,        // FP32 scalar
                  const bfloat16 *A,        // BF16 input
                  const int lda,
                  const bfloat16 *B,        // BF16 input
                  const int ldb,
                  const float beta,         // FP32 scalar
                  float *C,                 // FP32 output
                  const int ldc);
   
   } // namespace cosma
   ```

2. `src/cosma/blas.cpp`:
   Implement 3 backend options:
   
   **Option 1: MKL (if available)**:
   ```cpp
   #ifdef COSMA_WITH_MKL_BLAS
   void gemm_bf16(...) {
       cblas_gemm_bf16bf16f32(CblasColMajor, CblasNoTrans, CblasNoTrans,
                              M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
   }
   #endif
   ```
   
   **Option 2: oneDNN (if available)**:
   ```cpp
   #ifdef COSMA_WITH_ONEDNN
   void gemm_bf16(...) {
       // Use dnnl_sgemm with bf16 data type
       // Requires creating oneDNN memory descriptors
   }
   #endif
   ```
   
   **Option 3: Fallback - Convert to FP32**:
   ```cpp
   void gemm_bf16(...) {
       // Convert BF16 → FP32, call cblas_sgemm, slower but universal
       std::vector<float> A_fp32(M * K);
       std::vector<float> B_fp32(K * N);
       
       // Convert BF16 to FP32
       for (size_t i = 0; i < M * K; ++i) A_fp32[i] = static_cast<float>(A[i]);
       for (size_t i = 0; i < K * N; ++i) B_fp32[i] = static_cast<float>(B[i]);
       
       cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                   M, N, K, alpha, A_fp32.data(), M, B_fp32.data(), K, 
                   beta, C, M);
   }
   ```

3. `src/cosma/local_multiply.hpp`:
   ```cpp
   // NEW: Specialized signature for BF16 × BF16 → FP32
   template <>
   void local_multiply<bfloat16>(cosma_context<bfloat16>* ctx,
                                  bfloat16 *a,
                                  bfloat16 *b,
                                  float *c,  // NOTE: FP32 output!
                                  int m, int n, int k,
                                  float alpha, float beta,
                                  bool copy_c_back);
   ```

4. `src/cosma/local_multiply.cpp`:
   - Add specialized template for BF16
   - Call `gemm_bf16` instead of generic `gemm`

### Phase 3: Mixed-Precision Matrix Type
**Goal**: Create a `MixedPrecisionMatrix` class or extend `CosmaMatrix` for BF16 data with FP32 output

**Challenge**: Current `CosmaMatrix<Scalar>` assumes uniform type. Need to represent:
- Storage: BF16
- Computation output: FP32

**Approach**:
```cpp
// NEW: Trait to define output type for mixed precision
template <typename InputScalar>
struct output_scalar {
    using type = InputScalar;  // Default: same type
};

template <>
struct output_scalar<bfloat16> {
    using type = float;  // BF16 → FP32
};

// Use in CosmaMatrix:
template <typename Scalar>
class CosmaMatrix {
    using OutputScalar = typename output_scalar<Scalar>::type;
    // ...
};
```

**Files to modify**:
1. `src/cosma/matrix.hpp`:
   - Add `output_scalar` trait
   - Update `CosmaMatrix` to support mixed precision

2. `src/cosma/matrix.cpp`:
   - Add template instantiation for `CosmaMatrix<bfloat16>`

### Phase 4: Context and Memory Pool
**Goal**: Support BF16 in memory allocation and context management

**Files to modify**:
1. `src/cosma/memory_pool.cpp`:
   ```cpp
   template class cosma::memory_pool<bfloat16>;
   ```

2. `src/cosma/context.cpp`:
   ```cpp
   template class cosma_context<bfloat16>;
   ```

3. `src/cosma/buffer.cpp`:
   ```cpp
   template class Buffer<bfloat16>;
   ```

### Phase 5: MPI Communication
**Goal**: Enable distributed operations with BF16 data

**Files to modify**:
1. `libs/COSTA/src/costa/grid2grid/mpi_type_wrapper.hpp` (already covered in Phase 1)

2. `src/cosma/communicator.cpp`:
   - Add template instantiations for BF16 communication operations
   ```cpp
   template void communicator::copy<bfloat16>(...);
   template void communicator::reduce<bfloat16>(...);
   ```

3. `src/cosma/two_sided_communicator.cpp`:
   - Add template instantiations
   ```cpp
   template void copy<bfloat16>(...);
   template void reduce<bfloat16>(...);
   ```

### Phase 6: High-Level API
**Goal**: Expose BF16 matmul through COSMA's public API

**Files to modify**:
1. `src/cosma/multiply.cpp`:
   - Add template instantiation for `multiply<bfloat16>`
   
2. `src/cosma/cosma_pxgemm.cpp`:
   - Add BF16 variant of `pxgemm`

3. `src/cosma/pxgemm.cpp`:
   - Add explicit instantiation for BF16

## Testing Strategy

### Unit Tests
1. **Type conversion tests** (`tests/test_bfloat16.cpp`):
   - BF16 ↔ FP32 conversion accuracy
   - Edge cases (NaN, Inf, denormals)

2. **Local multiply tests** (`tests/test_bf16_local_multiply.cpp`):
   - Small matrix multiply: BF16 × BF16 → FP32
   - Compare against FP32 × FP32 → FP32 (should be close)
   - Verify accumulation is in FP32

3. **Distributed multiply tests** (`tests/test_bf16_distributed.cpp`):
   - Multi-rank BF16 matmul
   - Compare distributed vs single-rank result

### Integration Tests
1. **Compare backends**:
   - MKL BF16 vs fallback (if MKL available)
   - Ensure numerical agreement within tolerance

2. **Performance benchmarks**:
   - BF16 vs FP32 throughput
   - Memory bandwidth savings (BF16 is 50% of FP32)

### Accuracy Tests
- **Expected precision**: BF16 has ~3 decimal digits of precision
- **Tolerance**: Use relative error ~1e-2 to 1e-3 for correctness tests
- **Comparison**: BF16 result should match FP32 within ~0.1-1% relative error

## Implementation Phases

### MVP (Minimum Viable Product) - Phase 1
**Goal**: Get BF16 matmul working with fallback implementation

1. ✅ Define `bfloat16` type
2. ✅ Implement `gemm_bf16` with FP32 fallback
3. ✅ Add template instantiations
4. ✅ Write basic unit test
5. ✅ Verify single-rank multiply works

**Estimated effort**: 2-3 days

### Phase 2: Optimized Backend
**Goal**: Integrate MKL or oneDNN for native BF16 performance

1. Add conditional compilation for MKL BF16
2. Add oneDNN integration (optional)
3. Benchmark: BF16 (fallback) vs BF16 (MKL) vs FP32

**Estimated effort**: 2-3 days

### Phase 3: Distributed Support
**Goal**: Enable multi-rank BF16 operations

1. MPI communication support
2. Grid2grid transformations for BF16
3. Distributed correctness tests

**Estimated effort**: 3-4 days

### Phase 4: Production Readiness
**Goal**: Documentation, testing, CI integration

1. Comprehensive test suite
2. Documentation updates
3. CMake integration (detect MKL BF16 support)
4. CI pipeline

**Estimated effort**: 2-3 days

## Open Questions

1. **MPI reduce operations**: How to handle sum reduction with BF16? 
   - Option A: Create custom MPI_Op for BF16 sum
   - Option B: Convert to FP32, reduce, convert back (slower but simpler)

2. **GPU support**: Should we support BF16 on GPUs in this PR?
   - CUDA has native `__nv_bfloat16`
   - ROCm has `hip_bfloat16`
   - Defer to future PR?

3. **Transpose operations**: Does BF16 need special handling for transpose?
   - Likely no, just copy operations

4. **Storage format**: Should BF16 matrices use optimized layout?
   - Initial implementation: same layout as FP32
   - Future: explore packed BF16 for cache efficiency

## Dependencies

### Required
- C++17 or later (for better type traits)
- CMake 3.14+

### Optional
- Intel MKL 2020+ (for `cblas_gemm_bf16bf16f32`)
- oneDNN (for `dnnl_sgemm` with BF16)
- CUDA 11+ (for GPU BF16 support)
- ROCm 4.5+ (for AMD GPU BF16 support)

## Backwards Compatibility

✅ **No breaking changes expected**:
- All new functionality
- Existing FP32/FP64 paths unchanged
- New BF16 type is additive

## Performance Considerations

### Memory Bandwidth Savings
- BF16: 2 bytes per element
- FP32: 4 bytes per element
- **50% reduction** in memory traffic for A and B matrices
- C matrix still FP32 (no savings)

### Compute Performance
- **With MKL/oneDNN**: Near-native BF16 performance (2-4× faster than FP32 on modern CPUs)
- **Fallback**: Slower than FP32 due to conversion overhead

### Recommended Use Cases
- Large matrix multiplications (M, N, K > 512)
- Memory-bound workloads
- Acceptable precision loss (~0.1-1%)

## References
- [BFloat16 Spec (Brain Floating Point)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [Intel MKL BF16 GEMM](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/cblas-gemm-bf16bf16f32.html)
- [oneDNN BF16 Documentation](https://oneapi-src.github.io/oneDNN/dev_guide_data_types.html)
