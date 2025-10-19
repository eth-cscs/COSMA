/**
 * @file test_bfloat16_mpi.cpp
 * @brief BFloat16 MPI communication tests
 *
 * Tests BF16 data transfers across MPI ranks using cosma communicator
 * functions. This validates that BF16 can be transferred correctly using
 * MPI_UINT16_T.
 *
 * @author David Sanftenberg
 */

#include <cosma/bfloat16.hpp>
#include <cosma/communicator.hpp>
#include <cosma/interval.hpp>
#include <cosma/mpi_mapper.hpp>
#include <cosma/two_sided_communicator.hpp>
#include <mpi.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace cosma;

bool test_mpi_send_receive() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "ERROR: This test requires at least 2 MPI ranks"
                      << std::endl;
        }
        return false;
    }

    const int N = 16;
    std::vector<bfloat16> send_buffer(N);
    std::vector<bfloat16> recv_buffer(N);

    // Rank 0 sends, Rank 1 receives
    if (rank == 0) {
        // Initialize with known values
        for (int i = 0; i < N; ++i) {
            send_buffer[i] = bfloat16(static_cast<float>(i + 1));
        }

        MPI_Send(send_buffer.data(), N, MPI_UINT16_T, 1, 0, MPI_COMM_WORLD);
        std::cout << "Rank 0: Sent " << N << " BF16 values to Rank 1"
                  << std::endl;
    } else if (rank == 1) {
        MPI_Recv(recv_buffer.data(),
                 N,
                 MPI_UINT16_T,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // Verify received data
        bool passed = true;
        for (int i = 0; i < N; ++i) {
            float expected = static_cast<float>(i + 1);
            float received = static_cast<float>(recv_buffer[i]);

            if (std::abs(received - expected) > 1e-6f) {
                std::cerr << "  recv_buffer[" << i << "] = " << received
                          << " (expected: " << expected << ")" << std::endl;
                passed = false;
            }
        }

        if (passed) {
            std::cout << "Rank 1: Successfully received and verified " << N
                      << " BF16 values" << std::endl;
        }
        return passed;
    }

    return true;
}

bool test_mpi_broadcast() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 8;
    std::vector<bfloat16> buffer(N);

    if (rank == 0) {
        // Root initializes data
        for (int i = 0; i < N; ++i) {
            buffer[i] = bfloat16(static_cast<float>(i * 2 + 1));
        }
        std::cout << "Rank 0: Broadcasting " << N << " BF16 values"
                  << std::endl;
    }

    // Broadcast from rank 0 to all ranks
    MPI_Bcast(buffer.data(), N, MPI_UINT16_T, 0, MPI_COMM_WORLD);

    // All ranks verify
    bool passed = true;
    for (int i = 0; i < N; ++i) {
        float expected = static_cast<float>(i * 2 + 1);
        float received = static_cast<float>(buffer[i]);

        if (std::abs(received - expected) > 1e-6f) {
            std::cerr << "Rank " << rank << ": buffer[" << i
                      << "] = " << received << " (expected: " << expected << ")"
                      << std::endl;
            passed = false;
        }
    }

    if (passed && rank != 0) {
        std::cout << "Rank " << rank << ": Successfully received broadcast data"
                  << std::endl;
    }

    return passed;
}

bool test_mpi_allreduce() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4;
    std::vector<float> send_fp32(N);
    std::vector<float> recv_fp32(N);

    // Each rank contributes rank+1 to each element
    for (int i = 0; i < N; ++i) {
        send_fp32[i] = static_cast<float>(rank + 1);
    }

    // Perform Allreduce with FP32 (BF16 doesn't have MPI_SUM defined)
    MPI_Allreduce(send_fp32.data(),
                  recv_fp32.data(),
                  N,
                  MPI_FLOAT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    // Convert result to BF16 and back to verify precision
    std::vector<bfloat16> bf16_result(N);
    for (int i = 0; i < N; ++i) {
        bf16_result[i] = bfloat16(recv_fp32[i]);
    }

    // Verify
    float expected_sum = 0.0f;
    for (int r = 0; r < size; ++r) {
        expected_sum += static_cast<float>(r + 1);
    }

    bool passed = true;
    for (int i = 0; i < N; ++i) {
        float result = static_cast<float>(bf16_result[i]);
        // BF16 precision loss is acceptable for small integers
        if (std::abs(result - expected_sum) > 0.5f) {
            std::cerr << "Rank " << rank << ": result[" << i << "] = " << result
                      << " (expected: " << expected_sum << ")" << std::endl;
            passed = false;
        }
    }

    if (passed && rank == 0) {
        std::cout << "Allreduce test passed (sum across " << size
                  << " ranks = " << expected_sum << ")" << std::endl;
    }

    return passed;
}

bool test_mpi_type_mapper() {
    // Verify mpi_mapper returns correct MPI type for bfloat16
    MPI_Datatype bf16_type = mpi_mapper<bfloat16>::getType();

    if (bf16_type != MPI_UINT16_T) {
        std::cerr
            << "ERROR: mpi_mapper<bfloat16>::getType() returned wrong type"
            << std::endl;
        return false;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "MPI type mapper test passed (BF16 → MPI_UINT16_T)"
                  << std::endl;
    }

    return true;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "===== BFloat16 MPI Communication Tests ====="
                  << std::endl;
        std::cout << std::endl;
    }

    bool all_passed = true;

    // Test 1: MPI type mapper
    if (rank == 0)
        std::cout << "Testing MPI type mapper..." << std::endl;
    all_passed &= test_mpi_type_mapper();
    MPI_Barrier(MPI_COMM_WORLD);

    // Test 2: Send/Receive
    if (rank == 0)
        std::cout << "\nTesting MPI Send/Receive..." << std::endl;
    all_passed &= test_mpi_send_receive();
    MPI_Barrier(MPI_COMM_WORLD);

    // Test 3: Broadcast
    if (rank == 0)
        std::cout << "\nTesting MPI Broadcast..." << std::endl;
    all_passed &= test_mpi_broadcast();
    MPI_Barrier(MPI_COMM_WORLD);

    // Test 4: Allreduce (via FP32)
    if (rank == 0)
        std::cout << "\nTesting MPI Allreduce (via FP32)..." << std::endl;
    all_passed &= test_mpi_allreduce();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n======================================" << std::endl;
        if (all_passed) {
            std::cout << "All MPI tests passed!" << std::endl;
        } else {
            std::cout << "Some MPI tests FAILED!" << std::endl;
        }
    }

    MPI_Finalize();

    return all_passed ? 0 : 1;
}
