# COSMA: Communication-Optimal, S-partition based, Matrix-Multiplication Algorithm

## Overview

COSMA is a parallel, high-performance, GPU-accelerated, matrix-matrix
mutliplication algorithm that is communication-optimal for all combinations of
matrix dimensions, number of processes and memory sizes, without the need of any
parameter tuning. The key idea behind COSMA is to first derive a tight optimal
sequential schedule and only then parallelize it, preserving I/O optimality
between processes. This stands in contrast with the 2D and 3D algorithms, which
fix process domain decomposition upfront and then map it to the matrix
dimensions, which may result in asymptotically more communication. The final
design of COSMA facilitates the overlap of computation and communication,
ensuring speedups and applicability of modern mechanisms such as RDMA. COSMA
allows to not utilize some processors in order to optimize the processor grid,
which reduces the communication volume even further and increases the
computation volume per processor.

COSMA alleviates the issues of current state-of-the-art algorithms, which can be
summarized as follows:

- `2D (SUMMA)`: Requires manual tuning and not communication-optimal in the
  presence of extra memory.
- `2.5D`: Optimal for `m=n`, but inefficient for `m << n` or `n << m` and for
  some numbers of processes `p`.
- `Recursive (CARMA)`: Asymptotically communication-optimal for all `m, n, k,
  p`, but splitting always the largest dimension might lead up to `√3` increase
  in communication volume. 
- `COSMA (this work)`: Strictly communication-optimal (not just asymptotically)
  for all `m, n, k, p` and memory sizes that yields the speedups by factor of up
    to 8.3x over the second-fastest algorithm.

In addition to being communication-optimal, this implementation is
higly-optimized to reduce the memory footprint in the following sense:

- `Buffer Reuse`: all the buffers are pre-allocated and carefully reused during
  execution, including the buffers necessary for the communication, which
  reduces the total memory usage.
- `Reduced Local Data Movement`: the assignment of data blocks to processes is
  fully adapted to communication pattern, which minimizes the need of local data
  reshuffling that arise after each communication step.

The library supports both one-sided and two-sided MPI communication backends. It
uses `dgemm` for the local computations, but also has a support for the `GPU`
acceleration through our `Tiled-MM` library using `cublas` 


## Building

The project uses submodules, to clone do:

```bash
git clone --recursive https://github.com/eth-cscs/COSMA.git
```

> !! Note the *--recursive* flag !! 

COSMA is a CMake project, a recent CMake(>=3.12) is needed. From a build
directory, adjust and run `build.sh` found under the `scripts` directory.

External dependencies: 

- `MPI 3` (required)
- `Intel MKL` (default) or `CUDA` depending on whether CPU or GPU back end is
  used.

> Some dependencies are bundled as submodules and need not be installed
> explicitly:
>
> - `TiledMM` - cublasxt GEMM replacement
> - `grid2grid` - distributed matrix grid converter
> - `options` - command line utlility
> - `semiprof` - profiling utlility
> - `gtest_mpi` - MPI utlility wrapper over GoogleTest(unit testing library)

## Installing

To install do `make install`. 

> !! Note: To set custom installation directory use `CMAKE_INSTALL_PREFIX` when building. 

COSMA is CMake friendly and provides a cosmaConfig.cmake module for easy
integration into 3rd-party CMake projects with

```
find_package(cosma REQUIRED)
target_link_libraries( ... cosma::cosma)
```

COSMA's dependencies are taken care of internally, nothing else needs to be
linked. Make sure to set `CMAKE_PREFIX_PATH` to COSMA's installation directory
when building.

There is a rudimentary pkgconfig support; dependencies are handles explicitly by
consumers.

## Testing

To build all test targets:

```bash
make tests
```

To run all tests:

```bash
ctest
```

> !! Note: `COSMA_WITH_OPENMPI=ON` has to be set if OpenMPI is used.

## Miniapps

```bash
sbatch schedule_miniapp_on_daint.sh
```
The script will use SLURM to submit a job on 64 nodes. The job will run 2 matrix
multiplications and output the time COSMA algorithm took.

### Matrix Multiplication

The project contains a miniapp that produces two random matrices `A` and `B`,
computes their product `C` with the COSMA algorithm and outputs the time of the
multiplication.

The miniapp consists of an executable `./build/miniapp/cosma-miniapp` which can
be run with the following command line (assuming we are in the root folder of
the project):

```
n_iter=1 mpirun --oversubscribe -np 4 ./build/miniapp/cosma-miniapp -m 1000 -n 1000 -k 1000 -P 4 -s pm2,sm2,pk2
```

The flags have the following meaning:

- `-m (--m_dimension)`: number of rows of matrices `A` and `C`
- `-n (--n_dimension)`: number of columns of matrices `B` and `C`
- `-k (--k_dimension)`: number of columns of matrix `A` and rows of matrix `B`
- `-P (--processors)`: number of processors (i.e. ranks)
- `-s (--steps, optional)`: string of triplets divided by comma defining the
  splitting strategy. Each triplet defines one step of the algorithm. The first
  character in the triplet defines whether it is a parallel (p) or a sequential
  (s) step. The second character defines the dimension that is splitted in this
  step. The third parameter is an integer which defines the divisor. This
  parameter can be omitted. In that case the default strategy will be used.
- `-L (--memory, optional)`: memory limit, describes how many elements at most
  each rank can own. If not set, infinite memory will be assumed and the default
  strategy will only consist of parallel steps.
- `-t (--topology, optional)`: if this flag is present, then ranks might be
  relabelled such that the ranks which communicate are physically closer to each
  other. This flag therefore determines whether the topology is
  communication-aware.

### COSMA PDGEMM Wrapper

COSMA also contains a wrapper for ScaLAPACK `pxgemm` calls which offers scalapack interface (pxgemm functions with exactly the same signatures as ScaLAPACK). Running these functions will take care of transforming the matrices between ScaLAPACK and COSMA data layout, perform the multiplication using COSMA algorithm and transform the result back to the specified ScaLAPACK data layout.

The miniapp consists of an executable `./build/miniapp/pdgemm-miniapp` which can
be run with the following command line on Piz Daint (assuming we are in the root folder of
the project):

```
OMP_NUM_THREADS=18 MKL_NUM_THREADS=18 srun -C mc -N 8 -n 16 ./build/miniapp/pdgemm-miniapp -m 1000 -n 1000 -k 1000 -p 4 -q 4 -bm 128 -bn 128 -bk 128 --trans_a
```

The flags have the following meaning:

- `-m (--m_dim)`: number of rows of matrices `A` and `C`
- `-n (--n_dim)`: number of columns of matrices `B` and `C`
- `-k (--k_dim)`: number of columns of matrix `A` and rows of matrix `B`
- `-bm (--m_block)` (optional, default 128): block size for dimension m
- `-bn (--n_block)` (optional, default 128): block size for dimension n
- `-bk (--k_block)` (optional, default 128): block size for dimension k
- `-ta (--trans_a)` (optional, default: no transpose): transpose A before mutliplication
- `-tb (--trans_b)` (optional, default: no transpose): transpose B before mutliplication

### Dry-run for statistics

If interested in the communication or computation volume, maximum buffer size or
a maximum local matrix-multiplication size, you can use the dry-run, which
simulates the algorithm without actually performing any communication or
computation. This dry-run mode is not distributed, even though it simulates a
distributed version of COSMA.

The meaning of flags is the same as in previous examples. It can be run from the
project directory with: 

```
./build/miniapp/cosma-statistics -m 1000 -n 1000 -k 1000 -P 4 -s pm2,sm2,pk2
```

Executing the previous command produces the following output:

```
Matrix dimensions (m, n, k) = (1000, 1000, 1000)
Number of processors: 4
Divisions strategy:
parallel (m / 2)
sequential (m / 2)
parallel (k / 2)

Total communication units: 2000000
Total computation units: 125000000
Max buffer size: 500000
Local m = 250
Local n = 1000
Local k = 500
```

All the measurements are given in the units representing the number of elements
of the matrix (not in bytes).


## Profiling

Use `-DCOSMA_WITH_PROFILING=ON` to instrument the code. We use the profiler
called `semiprof`, written by Benjamin Cumming (https://github.com/bcumming).

Running the miniapp locally (from the project root folder) with the following
command:

```bash
mpirun --oversubscribe -np 4 ./build/miniapp/cosma-miniapp -m 1000 -n 1000 -k 1000 -P 4 -s pm2,sm2,pk2
```

Produces the following output from rank 0:

```
Matrix dimensions (m, n, k) = (1000, 1000, 1000)
Number of processors: 4
Divisions strategy:
parallel (m / 2)
sequential (m / 2)
parallel (k / 2)

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

The precentage is always relative to the first level above. All time
measurements are in seconds.

## Authors

- Grzegorz Kwasniewski, Marko Kabic, Maciej Besta, Raffaele Solca, Joost
  VandeVondele, Torsten Hoefler

## Questions?

For questions, feel free to contact us!
- For questions regarding theory, contact Grzegorz Kwasniewski (gkwasnie@inf.ethz.ch).
- For questions regarding the implementation, contact Marko Kabic (marko.kabic@cscs.ch).

## Ackowledgements

We thank Thibault Notargiacomo, Sam Yates, Benjamin Cumming and Simon Pintarelli
for their generous contribution to the project: great ideas, useful advices and 
fruitful discussions.

