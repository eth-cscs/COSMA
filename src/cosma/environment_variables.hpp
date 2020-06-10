#include <string>
#include <limits>
#include <complex>
#include <stdlib.h>
#include <limits>

namespace cosma {

// names of supported environment variables
namespace env_var_names {
    // number of GPU streams to be used per rank
    const std::string gpu_n_streams = "COSMA_GPU_STREAMS";
    // max sizes of GPU tiles (in #elements)
    // MxN corresponds to matrix C and K to the shared dimension
    const std::string gpu_tile_m = "COSMA_GPU_MAX_TILE_M";
    const std::string gpu_tile_n = "COSMA_GPU_MAX_TILE_N";
    const std::string gpu_tile_k = "COSMA_GPU_MAX_TILE_K";
    // if ON, COSMA will try to natively use scalapack layout
    // without transformation. Only used in the pxgemm wrapper.
    const std::string adapt_strategy = "COSMA_ADAPT_STRATEGY";
    // if ON, COSMA will try to overlap communication and computation
    const std::string overlap = "COSMA_OVERLAP_COMM_AND_COMP";
    // specifies the maximum available CPU memory per rank in MB
    const std::string cpu_max_memory = "COSMA_CPU_MAX_MEMORY";
    // if true, local host matrices will be pinned 
    // (only used when GPU backend enabled)
    // which increases the efficiency
    const std::string memory_pinning_enabled = "COSMA_GPU_MEMORY_PINNING";
};

// default values of supported environment variables
namespace env_var_defaults {
    // number of GPU streams to be used per rank
    const int gpu_n_streams = 2;
    // max sizes of GPU tiles (in #elements)
    // MxN corresponds to matrix C and K to the shared dimension
    const int gpu_tile_m = 5000;
    const int gpu_tile_n = 5000;
    const int gpu_tile_k = 5000;
    // if ON, COSMA will try to natively use scalapack layout
    // without transformation. Only used in the pxgemm wrapper.
    const bool adapt_strategy = true;
    // if ON, COSMA will try to overlap communication and computation
    const bool overlap = false;
    // specifies the maximum available CPU memory per rank in MB
    const long long cpu_max_memory = std::numeric_limits<long long>::max(); // inf
    // if true, local host matrices will be pinned 
    // (only used when GPU backend enabled)
    // which increases the efficiency
    const bool memory_pinning_enabled = true;
};

// checks if the specified environment variable is defined
bool env_var_defined(const char* var_name);

// checks if the environment variable with given name
// is set to ON or OFF. If the variable is not defined,
// the default value is returned
bool get_bool_env_var(std::string name, bool default_value);

// gets the value of the specified environment variable.
// If the variable is not defined, the default value is returned
int get_int_env_var(std::string name, int default_value);

// reads the environment variable corresponding to
// the number of GPU streams per rank and returns
// the default value if the variable is undefined
int gpu_streams();

// reads the environment variable corresponding to
// the maximum tile sizes on GPU and returns
// the default values if the variables are undefined.
// MxN corresponds to matrix C and K to the shared dimension
int gpu_max_tile_m();
int gpu_max_tile_n();
int gpu_max_tile_k();

// reads the environment variable corresponding to
// the adaptation of strategy and returns
// the default value if the variable is undefined
bool get_adapt_strategy();

// reads the environment variable corresponding to
// the overlap of communication and computation and returns
// the default value if the variable is undefined
bool get_overlap_comm_and_comp();

// reads the environment variable corresponding to
// the memory limit in MB per rank, converts the limit 
// to #elements that each rank is allowed to use.
// returns the default value if the variable is undefined
template <typename T>
long long get_cpu_max_memory();

bool get_memory_pinning();
}
