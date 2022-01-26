#pragma once

#ifdef COSMA_WITH_OFFLOAD_PROFILING
#include <roctracer/roctx.h>
// Start roctx/nvtx range
#define COSMA_RE(name) (roctxRangePush(#name))
// End roctx/nvtx range
#define COSMA_RL() (roctxRangePop())
#else
#define COSMA_RE(name)
#define COSMA_RL(name)
#endif


// The header makes semiprof an optional dependency that needs not be shipped when COSMA is installed.
//
#ifdef COSMA_WITH_PROFILING

//------------------------------------
#include <semiprof/semiprof.hpp>
// prints the profiler summary
#define PP() std::cout << semiprof::profiler_summary() << "\n"
// clears the profiler (counts and timings)
#define PC() semiprof::profiler_clear()

#ifdef COSMA_WITH_OFFLOAD_PROFILING
#define COSMA_PE(name) \
    { \
        COSMA_RE(name); \
        PE(name); \
    }
#define COSMA_PL() \
    { \
        COSMA_RL(); \
        PL(); \
    }
#else
#define COSMA_PE(name) (PE(name))
#define COSMA_PL() (PL())
#endif
//------------------------------------

#else

//------------------------------------
#define COSMA_PE(name)
#define COSMA_PL()
#define PP()
#define PC()
//------------------------------------

#endif
