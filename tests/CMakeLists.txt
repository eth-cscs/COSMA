# google test framework
add_library(gtest gtest-all.cpp)

# Unit tests
add_subdirectory(unit)
add_subdirectory(ubench)
