## Overview
CARMA (Demmel et al., 2013) is the first matrix-matrix multiplication algorithm that is communication-optimal for all memory ranges and all matrix shapes. [paper](http://www.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-205.pdf), [poster](http://www.cs.berkeley.edu/~odedsc/papers/CARMA%20Poster-SC12).

The algorithm recursively splits the largest matrix dimension creating smaller subproblems which are then recursively solved sequentially (DFS step) or in parallel (BFS step), depending on the available memory. When the base case of the recursion is reached, CARMA uses `dgemm` to perform the local multiplication. While appealing and simple at first sight, the implementation details are tricky and the distributed version requires the data layout very different from any layout used in existing linear-algebra libraries. Concretely, whenever the largest matrix is split, these two halves should already reside entirely on the corresponding halves of the processors. This data layout requirement applies recursively to the base case at which point any load balanced data layout can be used.

The original implementation of CARMA assumes that all the dimensions and the number of processors are the powers of 2 or that the number of processors always shares common divisors with the largest dimension in each step of the recursion. It also uses a cyclic data layout in the base case which requires that the data is locally completely reshuffled after each communication.

Here, we present the results from an implementation of CARMA that provides functionality not present in earlier published prototypes, namely the ability to deal with matrix dimensions and processor numbers that are not powers of two, and do not necessarily share common divisors. Furthermore, we derive a relatively simple underlying data layout, which preserves the communication-optimality of the algorithm, but requires less intermediate data reshufflings during execution, has improved memory access patterns and is potentially more compatible with existing linear algebra libraries.

## Using two remotes

```sh
git remote add both git@github.com:eth-cscs/CARMA.git
git remote set-url --add both git@gitlab.ethz.ch:kabicm/CARMA.git
```

## How to build for Piz Daint ?

CARMA uses `dgemm` for the local computation in the base case. A flag `-DCARMA_LAPACK_TYPE` determines which `dgemm` is used. It can have values: `MKL` or `openblas`. If equal to `MKL`, the environment variable `MKLROOT` will be used to find `MKL`. If equal to openblas, the environment variable `BLASROOT` will be used to find `openblas`.

```
module swap PrgEnv-cray PrgEnv-gnu

module load daint-mc
module load intel
module load CMake

export CRAYPE_LINK_TYPE=dynamic
export MPICH_NEMESIS_ASYNC_PROGRESS=MC
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_GNI_ASYNC_PROGRESS_TIMEOUT=0

cd build

CXX=CC CC=cc cmake
  -DCMAKE_BUILD_TYPE=Release \
  -fopenmp \
  -DCARMA_LAPACK_TYPE=MKL \
  ..
make -j 8
```

## How to test
In the build directory, do:
```
make test
```

## Miniapp

The project contains the miniapp that produces two random matrices `A` and `B`, computes their product `C` with the CARMA algorithm and outputs all three matrices into files names as: `<matrix><rank>.txt` (for example `A0.txt` for local data in rank `0` of matrix `A`). Each file is the list of triplets in the form `row column element`.

The miniapp consists of an executable `./build/miniapp/carma-miniapp` which can be run with the following command line (assuming we are in the root folder of the project):
### Example:
```
mpirun --oversubscribe -np 4 ./build/miniapp/carma-miniapp -m 1000 -n 1000 -k 1000 -r 3 -p bdb -d 211211112
```
The flags have the following meaning:

- `m`: number of rows of matrices `A` and `C`

- `n`: number of columns of matrices `B` and `C`

- `k`: number of columns of matrix `A` and rows of matrix `B`

- `r`: number of recursive steps

- `p`: string of length `r` that represents the type of each step (either *b* for *BFS* step or *d* for *DFS* step).

- `d`: division pattern of length `3 r`. An `i-`th triplet `xyz` represents the divisors of `m`, `n` and `k` repsectively in `i-`th step. Only one of `x`, `y` and `z` can be `>1` while other have to be `=1`.


## Profiling the code
Use `-DCARMA_WITH_PROFILING=ON` to instrument the code. Running miniapp with the following command:

```mpirun --oversubscribe -np 4 ./miniapp/carma-miniapp -m 1000 -n 1000 -k 1000 -r 3 -p bdb -d 211211112```

Produces the following output from each rank (here only the output from rank 0 shown):

```
Benchmarking 1000*1000*1000 multiplication using 4 processes
Division pattern is: bdb - 211211112
RANK 0
 --------------------------------------------------------------
|                           PROFILER                           |
 --------------------------------------------------------------
| region                          t [ms]       [%]       count |
 --------------------------------------------------------------
|-total                          107.000     100.0         -
    |-multiply                   101.000      94.4         -
        |-communication           42.000      41.6         -
            |-copying              9.000      21.4         1
            |-reduction           33.000      78.6         2
        |-computation             50.000      49.5         2
        |-layout-overhead          0.000       0.0        18
    |-preprocessing                6.000       5.6         -
        |-layout-init              0.000       0.0         3
        |-mapper-init              6.000     100.0         3

```
All the time measurements are given in milliseconds. The precentage is always relative to the first level above.
The difference between the node time and the sum of the nested node times is caused by the overhead of the profiler itself.

### Requirements
CARMA algorithm uses:
  - `MPI`
  - `OpenMP`
  - `dgemm` that is provided either through `MKL` (Intel Parallel Studio XE), or through `openblas`.

### Authors
Marko Kabic \
(marko.kabic@cscs.ch)

### Mentors/Supervisors
Professor Dr. Joost VandeVondele \
Dr. Raffaele Solcà
