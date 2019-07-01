#include <gtest/gtest.h>

int main(int argc, char **argv) {
    // init gtest
    ::testing::InitGoogleTest(&argc, argv);
    // run all tests
    return RUN_ALL_TESTS();
}
