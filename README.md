# COSMA: Communication-Optimal, S-partition based, Matrix-Multiplication Algorithm

## Overview
COSMA is a parallel matrix-matrix mutliplication algorithm that is communication-optimal for all combinations of matrix dimensions, number of processes and memory sizes, without the need of any parameter tuning. The key idea behind COSMA is to first derive a tight optimal sequential schedule and only then parallelize it, preserving I/O optimality between processes. This stands in contrast with the 2D and 3D algorithms, which fix process domain decomposition upfront and then map it to the matrix dimensions, which may result in up to a factor of `√3` increase in communication volume. The final design of COSMA facilitates the overlap of computation and communication, ensuring speedups and applicability of modern mechanisms such as RDMA.

COSMA alleviates the issues of the current state-of-the-art algorithms, which can be summarized as follows:
- `2D (SUMMA)`: Requires manual tuning and not communication optimal in the presence of extra memory.
- `2.5D`: Optimal for `m=n`, but inefficient for `m << n` or `n << m` and for some numbers of processes `p`.
- `Recursive (CARMA)`: Asymptotically optimal for all `m, n, k, p`, but splitting always the largest dimension might lead up to `√3` increase in communication volume. 
- `COSMA (this work)`: Strictly communication-optimal (not just asymptotically) for all `m, n, k, p` and memory sizes that yields the speedups by factor of up to 8.3x over the second-fastest algorithm.

In addition to being communication-optimal, this implementation is higly-optimized to reduce the memory footprint in the following sense:
- `Buffer Reuse`: all the buffers are pre-allocated and carefully reused during execution, including the buffers necessary for the communication, which reduces the total memory usage by up to `25\%` when compared to CARMA.
- `Reduced Local Data Movement`: the assignment of data blocks to processes is completely adapted to communication pattern, which minimizes the need of local data reshuffling that arise after each communication step.

The library supports both one-sided and two-sided MPI communication backends. It uses `dgemm` for the local computations, but also has a support for the `GPU` acceleration throught `CUDA` 

## How to build for Piz Daint ?
A flag `-DCOSMA_LAPACK_TYPE` determines which `dgemm` is used. It can have values: `MKL` or `openblas`. If equal to `MKL`, the environment variable `MKLROOT` will be used to find `MKL`. If equal to openblas, the environment variable `BLASROOT` will be used to find `openblas`. If MKL is used, then the type of threading can be specified using the variable `MKL_THREADING` (values `Intel OpenMP`, `GNU OpenMP`, `Sequential`).

### Example (on Piz Daint)
```bash
# clone the repository
git clone https://github.com/eth-cscs/COSMA.git
cd COSMA

# setup the environment in order to use GNU compilers
# this is usually required at the cluster
# but not on a desktop system
module swap PrgEnv-cray PrgEnv-gnu

# load multicore partition
module load daint-mc
# load MKL from intel
module load intel
module load CMake

# enable the dynamic linking and
# the asynchronous thread progressing
# MPICH (on Cray systems)
export CRAYPE_LINK_TYPE=dynamic
export MPICH_NEMESIS_ASYNC_PROGRESS=MC
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_GNI_ASYNC_PROGRESS_TIMEOUT=0

# setup the right compilers
export CC=`which cc`
export CXX=`which CC`

# build the main project
mkdir build
cd build

cmake
  -DCMAKE_BUILD_TYPE=Release \
  -DCOSMA_LAPACK_TYPE=MKL \
  -DMKL_THREADING="Sequential" \
  ..
make -j 4
```

## How to test
In the build directory, do:
```bash
make test
```


## Miniapp
The project contains the miniapp that produces two random matrices `A` and `B`, computes their product `C` with the COSMA algorithm and outputs all three matrices into files names as: `<matrix><rank>.txt` (for example `A0.txt` for local data in rank `0` of matrix `A`). Each file is the list of triplets in the form `row column element`.

The miniapp consists of an executable `./build/miniapp/cosma-miniapp` which can be run with the following command line (assuming we are in the root folder of the project):

### Example:
```
mpirun --oversubscribe -np 4 ./build/miniapp/cosma-miniapp -m 1000 -n 1000 -k 1000 -P 4 -s bm2,dm2,bk2
```
The flags have the following meaning:

- `-m (--m_dimension)`: number of rows of matrices `A` and `C`

- `-n (--n_dimension)`: number of columns of matrices `B` and `C`

- `-k (--k_dimension)`: number of columns of matrix `A` and rows of matrix `B`

- `-P (--processors)`: number of processors (i.e. ranks)

- `-s (--steps, optional)`: string of triplets divided by comma defining the splitting strategy. Each triplet defines one step of the algorithm. The first character in the triplet defines whether it is BFS (b) or DFS (d) step. The second character defines the dimension that is splitted in this step. The third parameter is an integer which defines the divisor. This parameter can be omitted. In that case the default strategy will be used.

- `-L (--memory, optional)`: memory limit, describes how many elements at most each rank can own. If not set, infinite memory will be assumed and the default strategy will only consist of parallel (BFS) steps.

- `-t (--topology, optional)`: if this flag is present, then ranks might be relabelled such that the ranks which communicate are physically closer to each other. This flag therefore determines whether the topology is communication-aware.

In addition to this miniapp, after compilation, in the same directory (./build/miniapp/) there will be an executable called `cosma-statistics` which simulates the algorithm (without actually computing the matrix multiplication) in order to get the total volume of the communication, the maximum volume of computation done in a single branch and the maximum required buffer size that the algorithm requires.

### Example:
```
./build/miniapp/cosma-statistics -m 1000 -n 1000 -k 1000 -P 4 -s bm2,dm2,bk2
```
Executing the previous command produces the following output:

```
Matrix dimensions (m, n, k) = (1000, 1000, 1000)
Number of processors: 4
Divisions strategy:
BFS (m / 2)
DFS (m / 2)
BFS (k / 2)

Total communication units: 2000000
Total computation units: 125000000
Max buffer size: 500000
Local m = 250
Local n = 1000
Local k = 500
```
All the measurements are given in the units representing the number of elements of the matrix (not in bytes).


## Profiling the code
Use `-DCOSMA_WITH_PROFILING=ON` to instrument the code. We use the profiler called `semiprof`, written by Benjamin Cumming (https://github.com/bcumming).

### Example
Running the miniapp locally (from the project root folder) with the following command:

```bash
mpirun --oversubscribe -np 4 ./build/miniapp/cosma-miniapp -m 1000 -n 1000 -k 1000 -P 4 -s bm2,dm2,bk2 -d 211211112
```

Produces the following output from rank 0:

```
Matrix dimensions (m, n, k) = (1000, 1000, 1000)
Number of processors: 4
Divisions strategy:
BFS (m / 2)
DFS (m / 2)
BFS (k / 2)

_p_ REGION                     CALLS      THREAD        WALL       %
_p_ total                          -       0.110       0.110   100.0
_p_   multiply                     -       0.098       0.098    88.7
_p_     computation                2       0.052       0.052    47.1
_p_     communication              -       0.046       0.046    41.6
_p_       copy                     3       0.037       0.037    33.2
_p_       reduce                   3       0.009       0.009     8.3
_p_     layout                    18       0.000       0.000     0.0
_p_   preprocessing                3       0.012       0.012    11.3
```
The precentage is always relative to the first level above. All time measurements are in seconds.


### Requirements
COSMA algorithm uses:
  - `MPI`
  - `OpenMP`
  - `dgemm` that is provided either through `MKL` (Intel Parallel Studio XE), or through `openblas`.


### Authors
- *Theory:* Grzegorz Kwasniewski, Maciej Besta, Prof. Dr. Torsten Hoefler
- *Implementation:* Marko Kabic, Dr. Joost VandeVondele

### Ackowledgements
We thank Dr. Raffaele Solcà and Dr. Thibault Notargiacomo for generous help during the implementation.
