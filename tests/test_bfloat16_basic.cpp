/**
 * @file test_bfloat16_basic.cpp
 * @brief Basic unit tests for bfloat16 type and BF16 GEMM
 * @author David Sanftenberg
 * @date 2025-10-19
 */

#include <cosma/bfloat16.hpp>
#include <cosma/blas.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace cosma;

void test_bf16_conversion() {
    std::cout << "Testing BF16 ↔ FP32 conversion..." << std::endl;

    // Test simple values
    {
        float val = 1.0f;
        bfloat16 bf(val);
        float result = static_cast<float>(bf);
        assert(std::abs(result - val) < 1e-6f);
        std::cout << "  1.0f: " << result << " ✓" << std::endl;
    }

    {
        float val = 3.14159f;
        bfloat16 bf(val);
        float result = static_cast<float>(bf);
        // BF16 has ~3 decimal digits of precision
        assert(std::abs(result - val) / val < 0.01f); // 1% relative error
        std::cout << "  π: " << val << " → " << result
                  << " (error: " << std::abs(result - val) << ") ✓"
                  << std::endl;
    }

    {
        float val = -42.5f;
        bfloat16 bf(val);
        float result = static_cast<float>(bf);
        assert(std::abs(result - val) < 0.1f);
        std::cout << "  -42.5f: " << result << " ✓" << std::endl;
    }

    {
        float val = 0.0f;
        bfloat16 bf(val);
        float result = static_cast<float>(bf);
        assert(result == 0.0f);
        std::cout << "  0.0f: " << result << " ✓" << std::endl;
    }

    std::cout << "BF16 conversion tests passed!\n" << std::endl;
}

void test_bf16_arithmetic() {
    std::cout << "Testing BF16 arithmetic..." << std::endl;

    bfloat16 a(2.0f);
    bfloat16 b(3.0f);

    bfloat16 sum = a + b;
    assert(std::abs(static_cast<float>(sum) - 5.0f) < 1e-6f);
    std::cout << "  2 + 3 = " << static_cast<float>(sum) << " ✓" << std::endl;

    bfloat16 diff = a - b;
    assert(std::abs(static_cast<float>(diff) + 1.0f) < 1e-6f);
    std::cout << "  2 - 3 = " << static_cast<float>(diff) << " ✓" << std::endl;

    bfloat16 prod = a * b;
    assert(std::abs(static_cast<float>(prod) - 6.0f) < 1e-6f);
    std::cout << "  2 * 3 = " << static_cast<float>(prod) << " ✓" << std::endl;

    bfloat16 quot = b / a;
    assert(std::abs(static_cast<float>(quot) - 1.5f) < 0.01f);
    std::cout << "  3 / 2 = " << static_cast<float>(quot) << " ✓" << std::endl;

    std::cout << "BF16 arithmetic tests passed!\n" << std::endl;
}

void test_bf16_gemm_simple() {
#if defined(COSMA_WITH_MKL_BLAS) || defined(COSMA_WITH_BLIS_BLAS) ||           \
    defined(COSMA_WITH_BLAS)
    std::cout << "Testing BF16 GEMM (2×2 matrix multiply)..." << std::endl;

    // Simple 2×2 matrix multiply: C = A * B
    // A = [1 2]    B = [5 6]    C = [19 22]
    //     [3 4]        [7 8]        [43 50]

    const int M = 2, N = 2, K = 2;

    // Input matrices in BF16
    std::vector<bfloat16> A(M * K);
    std::vector<bfloat16> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    // Initialize A (column-major)
    A[0] = bfloat16(1.0f);
    A[1] = bfloat16(3.0f); // First column
    A[2] = bfloat16(2.0f);
    A[3] = bfloat16(4.0f); // Second column

    // Initialize B (column-major)
    B[0] = bfloat16(5.0f);
    B[1] = bfloat16(7.0f); // First column
    B[2] = bfloat16(6.0f);
    B[3] = bfloat16(8.0f); // Second column

    // Expected result (column-major)
    float expected[4] = {19.0f, 43.0f, 22.0f, 50.0f};

    // Call BF16 GEMM: C = 1.0 * A * B + 0.0 * C
    gemm_bf16(M, N, K, 1.0f, A.data(), M, B.data(), K, 0.0f, C.data(), M);

    // Verify results
    bool passed = true;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(C[i] - expected[i]);
        float rel_error = error / std::abs(expected[i]);

        std::cout << "  C[" << i << "] = " << C[i]
                  << " (expected: " << expected[i] << ", error: " << error
                  << ")" << std::endl;

        // Allow for BF16 precision loss (~1% relative error)
        if (rel_error > 0.02f) { // 2% tolerance
            std::cerr << "ERROR: Result " << i << " exceeds tolerance!"
                      << std::endl;
            passed = false;
        }
    }

    assert(passed);
    std::cout << "BF16 GEMM simple test passed!\n" << std::endl;
#else
    std::cout << "Skipping BF16 GEMM test (BLAS not available)\n" << std::endl;
#endif
}

void test_bf16_gemm_larger() {
#if defined(COSMA_WITH_MKL_BLAS) || defined(COSMA_WITH_BLIS_BLAS) ||           \
    defined(COSMA_WITH_BLAS)
    std::cout << "Testing BF16 GEMM (larger 4×4 matrix)..." << std::endl;

    const int M = 4, N = 4, K = 4;

    std::vector<bfloat16> A(M * K);
    std::vector<bfloat16> B(K * N);
    std::vector<float> C_bf16(M * N, 0.0f);
    std::vector<float> C_fp32(M * N, 0.0f);

    // Initialize with random-ish values
    for (int i = 0; i < M * K; ++i) {
        float val = static_cast<float>(i % 10) / 10.0f;
        A[i] = bfloat16(val);
    }

    for (int i = 0; i < K * N; ++i) {
        float val = static_cast<float>((i * 3) % 10) / 10.0f;
        B[i] = bfloat16(val);
    }

    // Compute with BF16 GEMM
    gemm_bf16(M, N, K, 1.0f, A.data(), M, B.data(), K, 0.0f, C_bf16.data(), M);

    // Compute reference with FP32
    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(K * N);

    for (int i = 0; i < M * K; ++i) {
        A_fp32[i] = static_cast<float>(A[i]);
    }

    for (int i = 0; i < K * N; ++i) {
        B_fp32[i] = static_cast<float>(B[i]);
    }

    gemm(M,
         N,
         K,
         1.0f,
         A_fp32.data(),
         M,
         B_fp32.data(),
         K,
         0.0f,
         C_fp32.data(),
         M);

    // Compare results
    float max_rel_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(C_bf16[i] - C_fp32[i]);
        float rel_error = error / (std::abs(C_fp32[i]) + 1e-8f);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    std::cout << "  Max relative error: " << max_rel_error << std::endl;
    assert(max_rel_error < 0.05f); // 5% tolerance for BF16

    std::cout << "BF16 GEMM larger test passed!\n" << std::endl;
#else
    std::cout << "Skipping larger BF16 GEMM test (BLAS not available)\n"
              << std::endl;
#endif
}

int main() {
    std::cout << "===== BFloat16 Basic Tests =====" << std::endl << std::endl;

    test_bf16_conversion();
    test_bf16_arithmetic();
    test_bf16_gemm_simple();
    test_bf16_gemm_larger();

    std::cout << "===== All tests passed! =====" << std::endl;

    return 0;
}
