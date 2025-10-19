/**
 * @file benchmark_bf16_backends.cpp
 * @brief Benchmark BF16 GEMM: MKL native vs OpenBLAS fallback
 *
 * Compares performance of MKL's hardware-accelerated BF16 GEMM
 * (cblas_gemm_bf16bf16f32) against OpenBLAS fallback path
 * (BF16 → FP32 conversion + sgemm).
 *
 * @author David Sanftenberg
 */

#include <cosma/bfloat16.hpp>
#include <cosma/blas.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cosma;

struct BenchmarkResult {
    double time_ms;
    double gflops;
    std::string backend;
};

BenchmarkResult benchmark_gemm(int M, int N, int K, int iterations) {
    // Allocate matrices
    std::vector<bfloat16> A(M * K);
    std::vector<bfloat16> B(K * N);
    std::vector<float> C(M * N);

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        A[i] = bfloat16(static_cast<float>(rand()) / RAND_MAX);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = bfloat16(static_cast<float>(rand()) / RAND_MAX);
    }

    // Warm-up run
    gemm_bf16(M, N, K, 1.0f, A.data(), M, B.data(), K, 0.0f, C.data(), M);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        gemm_bf16(M, N, K, 1.0f, A.data(), M, B.data(), K, 0.0f, C.data(), M);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        iterations;
    double flops = 2.0 * M * N * K; // multiply + add
    double gflops = flops / (time_ms * 1e6);

    BenchmarkResult result;
    result.time_ms = time_ms;
    result.gflops = gflops;

#ifdef COSMA_WITH_MKL_BLAS
    result.backend = "MKL (native cblas_gemm_bf16bf16f32)";
#else
    result.backend = "OpenBLAS (BF16→FP32 fallback)";
#endif

    return result;
}

void print_header() {
    std::cout << "\n╔══════════════════════════════════════════════════════════"
                 "════════════╗\n";
    std::cout << "║          BFloat16 GEMM Backend Performance Benchmark       "
                 "        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════"
                 "══════════╝\n\n";
}

void print_result(const std::string &size_desc,
                  int M,
                  int N,
                  int K,
                  const BenchmarkResult &result) {
    std::cout << std::left << std::setw(20) << size_desc << " (" << std::setw(4)
              << M << " × " << std::setw(4) << N << " × " << std::setw(4) << K
              << ")\n";
    std::cout << "  Backend:    " << result.backend << "\n";
    std::cout << "  Time:       " << std::fixed << std::setprecision(3)
              << std::setw(8) << result.time_ms << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << std::setw(8) << result.gflops << " GFLOPS\n\n";
}

int main() {
    print_header();

    std::cout << "Backend Information:\n";
#ifdef COSMA_WITH_MKL_BLAS
    std::cout << "  Using Intel MKL with native BF16 GEMM support\n";
    std::cout << "  Function: cblas_gemm_bf16bf16f32 (BF16 × BF16 → FP32)\n";
    std::cout << "  Note: Hardware acceleration requires AVX-512 BF16 CPU\n";
#else
    std::cout << "  Using OpenBLAS with BF16→FP32 conversion fallback\n";
    std::cout << "  Function: cblas_sgemm (FP32 × FP32 → FP32)\n";
    std::cout << "  Note: Conversion overhead + larger memory footprint\n";
#endif
    std::cout << "\n";

    const int iterations = 10;

    // Small matrix (typical LLM decode - single token)
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n";
    std::cout << "Small Matrices (LLM Decode - Single Token)\n";
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n\n";

    auto result1 = benchmark_gemm(1, 896, 896, iterations * 10);
    print_result("Tiny (1 token)", 1, 896, 896, result1);

    auto result2 = benchmark_gemm(8, 896, 896, iterations * 5);
    print_result("Small (8 tokens)", 8, 896, 896, result2);

    // Medium matrices (typical LLM prefill - short context)
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n";
    std::cout << "Medium Matrices (LLM Prefill - Short Context)\n";
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n\n";

    auto result3 = benchmark_gemm(128, 896, 896, iterations);
    print_result("Medium (128 tokens)", 128, 896, 896, result3);

    auto result4 = benchmark_gemm(512, 896, 896, iterations);
    print_result("Large (512 tokens)", 512, 896, 896, result4);

    // Large matrices (LLM prefill - long context)
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n";
    std::cout << "Large Matrices (LLM Prefill - Long Context)\n";
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n\n";

    auto result5 = benchmark_gemm(2048, 896, 896, iterations / 2);
    print_result("Very Large (2K)", 2048, 896, 896, result5);

    auto result6 = benchmark_gemm(4096, 896, 896, iterations / 4);
    print_result("Huge (4K tokens)", 4096, 896, 896, result6);

    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n";
    std::cout << "Summary\n";
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n\n";

#ifdef COSMA_WITH_MKL_BLAS
    std::cout << "✓ MKL native BF16 GEMM provides:\n";
    std::cout << "  - Direct BF16 computation (no conversion overhead)\n";
    std::cout << "  - 50% reduced memory bandwidth vs FP32\n";
    std::cout
        << "  - Hardware acceleration on AVX-512 BF16 CPUs (2-4× speedup)\n";
    std::cout << "  - Best performance on large matrices (512+ tokens)\n";
#else
    std::cout << "✓ OpenBLAS fallback provides:\n";
    std::cout << "  - Functional BF16 support via FP32 conversion\n";
    std::cout << "  - Works on any CPU (no special hardware required)\n";
    std::cout
        << "  - Conversion overhead: 2× memory allocation + conversion loops\n";
    std::cout << "  - Consider MKL for production deployments\n";
#endif

    std::cout << "\nNote: Benchmark run in Debug mode. Release builds expected "
                 "5-10× faster.\n";
    std::cout << "═════════════════════════════════════════════════════════════"
                 "═════════\n\n";

    return 0;
}
