#pragma once
#include <complex>
#include <limits>
#include <stdlib.h>
#include <string>

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
// The scaling factor used for the memory-pool allocation size.(cpu-only).
// If amortization = 1.2, then the memory allocator
// will request 1.2x the requested size (thus, 20% more than needed).
// Higher values better amortize the cost of memory buffers resizing
// which can occur when the algorithm is invoked for different matrix sizes.
// However, higher amortization values also mean that
// potentially more memory is allocated than used which can be
// a problem when the memory resource is tight.
// There is just a single memory pool in COSMA and all the required
// memory is taken from this memory pool only.
const std::string memory_pool_amortization = "COSMA_MEMORY_POOL_AMORTIZATION";
// minimum local matrix size -- if P is too large, so that after
// splitting the local matrix size get lower than this,
// then P will be reduced so that the problem size
// never gets smaller than specified by this variable
const std::string min_local_dimension = "COSMA_MIN_LOCAL_DIMENSION";
// if any dimension is smaller than this threshold, it will be dispatched to
// SCALAPACK since it's too "thin" for COSMA in that case
const std::string cosma_dim_threshold = "COSMA_DIM_THRESHOLD";
// number of bytes to which all host buffers are aligned
const std::string cosma_cpu_memory_alignment = "COSMA_CPU_MEMORY_ALIGNMENT";
// IF ON, use unified memory
const std::string cosma_gpu_unified_memory = "COSMA_GPU_UNIFIED_MEMORY";
}; // namespace env_var_names

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
// The scaling factor used for the memory-pool allocation size.(cpu-only).
// If amortization = 1.2, then the memory allocator
// will request 1.2x the requested size (thus, 20% more than needed).
// Higher values better amortize the cost of memory buffers resizing
// which can occur when the algorithm is invoked for different matrix sizes.
// However, higher amortization values also mean that
// potentially more memory is allocated than used which can be
// a problem when the memory resource is tight.
// There is just a single memory pool in COSMA and all the required
// memory is taken from this memory pool only.
const double memory_pool_amortization = 1.2;
// minimum local matrix size -- if P is too large, so that after
// splitting the local matrix size get lower than this,
// then P will be reduced so that the problem size
// never gets smaller than specified by this variable
const int min_local_dimension = 200;
// if any dimension is smaller than this threshold, it will be dispatched to
// SCALAPACK since it's too "thin" for COSMA in that case
const int cosma_dim_threshold = 0;
// cpu memory alignment (currently disabled)
const int cosma_cpu_memory_alignment = 0; // 256;
// gpu unified memory mechanism
const bool unified_memory = false;
}; // namespace env_var_defaults

// checks if the specified environment variable is defined
bool env_var_defined(const char *var_name);

// checks if the environment variable with given name
// is set to ON or OFF. If the variable is not defined,
// the default value is returned
bool get_bool_env_var(std::string name, bool default_value);

// gets the value of the specified environment variable.
// If the variable is not defined, the default value is returned
int get_int_env_var(std::string name, int default_value);

// gets the value of the specified environment variable.
// If the variable is not defined, the default value is returned
size_t get_ull_env_var(std::string name, size_t default_value);

// gets the value of the specified environment variable.
// If the variable is not defined, the default value is returned
float get_float_env_var(std::string name, float default_value);

// gets the value of the specified environment variable.
// If the variable is not defined, the default value is returned
double get_double_env_var(std::string name, double default_value);

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

// reads the memory pool amortization (>= 1.0).
// If amortization = 1.2, then the memory allocator
// will request 1.2x the requested size (thus, 20% more than needed).
// Higher values better amortize the cost of memory buffers resizing
// which can occur when the algorithm is invoked for different matrix sizes.
// However, higher amortization values also mean that
// potentially more memory is allocated than used which can be
// a problem when the memory resource is tight.
double get_memory_pool_amortization();

// reads the environment variable corresponding to
// the memory limit in MB per rank, converts the limit
// to #elements that each rank is allowed to use.
// returns the default value if the variable is undefined
template <typename T>
long long get_cpu_max_memory();

// whether host matrix buffers should be pinned or not
// this is only used in the GPU backend to increase
// the transfer speed between CPU and GPU
bool get_memory_pinning();

// if, after the matrices are split among ranks,
// any dimension becomes less than this threashold,
// then the total number of ranks is going to be reduced
// so that no dimension gets less than this threshold
// after splitting.
int get_min_local_dimension();

// if initial dimension (before splitting) is less
// than this threshold, the problem is considered too small
// and is dispatched to SCALAPACK
// This is only used for pxgemm wrappers.
int get_cosma_dim_threshold();

// number of bytes to which all the buffers should be aligned
int get_cosma_cpu_memory_alignment();

// check if we use unified memory or not
bool get_unified_memory();
} // namespace cosma
