#pragma once

// The header makes semiprof an optional dependency that needs not be shipped when COSMA is installed.
//
#ifdef COSMA_WITH_PROFILING

#include <semiprof/semiprof.hpp>

// prints the profiler summary
#define PP() std::cout << semiprof::profiler_summary() << "\n"
// clears the profiler (counts and timings)
#define PC() semiprof::profiler_clear()

#else
#define PE(name)
#define PL()
#define PP()
#define PC()
#endif
