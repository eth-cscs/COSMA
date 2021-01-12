find_package(OpenMP REQUIRED)
add_executable(omp omp.cpp)
target_link_libraries(omp PRIVATE OpenMP::OpenMP_CXX semiprof)
