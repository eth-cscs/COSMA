#include <cosma/memory_pool.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <vector>

using namespace cosma;

// Requesting/releasing memory should not require any MPI ranks or
// communication, so this is tested as a plain (non-MPI) gtest, similar
// to test.mapper.

TEST(memory_pool, free_releases_capacity) {
    memory_pool<double> pool;
    pool.amortization = 1.0;

    std::vector<size_t> buffer_sizes = {1000, 2000, 3000};
    pool.reserve(buffer_sizes);

    EXPECT_GT(pool.capacity(), 0u);

    pool.free();

    EXPECT_EQ(pool.capacity(), 0u);
    EXPECT_EQ(pool.size(), 0u);
}

TEST(memory_pool, usable_after_free) {
    memory_pool<double> pool;
    pool.amortization = 1.0;

    std::vector<size_t> buffer_sizes = {500};
    pool.reserve(buffer_sizes);
    pool.free();

    // reserving/using the pool again after free() should work exactly
    // as it would on a freshly-constructed pool.
    std::vector<size_t> buffer_sizes_2 = {500};
    EXPECT_NO_THROW(pool.reserve(buffer_sizes_2));
    EXPECT_NO_THROW(auto id = pool.get_buffer_id(500); pool.get_buffer_pointer(id));
}

TEST(memory_pool, reset_keeps_capacity_free_does_not) {
    memory_pool<double> pool;
    pool.amortization = 1.0;

    std::vector<size_t> buffer_sizes = {10000};
    pool.reserve(buffer_sizes);
    size_t reserved_capacity = pool.capacity();
    ASSERT_GT(reserved_capacity, 0u);

    pool.reset();
    // reset() only marks the pool as logically empty, capacity is unchanged
    EXPECT_EQ(pool.capacity(), reserved_capacity);

    pool.free();
    // free() actually gives the memory back
    EXPECT_EQ(pool.capacity(), 0u);
}

TEST(memory_pool, oversized_request_throws) {
    memory_pool<double> pool;
    pool.amortization = 1.0;

    // a request this large cannot be satisfied on any real machine and
    // should raise a length_error/bad_alloc (caught internally, message
    // printed, and rethrown), rather than silently corrupting state.
    std::vector<size_t> buffer_sizes = {
        std::numeric_limits<size_t>::max() / 2};
    EXPECT_ANY_THROW(pool.reserve(buffer_sizes));
}
