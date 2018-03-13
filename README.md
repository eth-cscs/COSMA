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

## Profiling the code
Use `-DCARMA_WITH_PROFILING=ON` to instrument the code. Running miniapp with the following command:

```mpirun --oversubscribe -np 4 ./miniapp/carma-miniapp -m 1000 -n 1000 -k 1000 -r 3 -p bdb -d 211211112```

Produces the following output:

```
Benchmarking 1000*1000*1000 multiplication using 4 processes
Division pattern is: bdb - 211211112
RANK 0
PROFILING RESULTS:
|- multiply: 134
    |- communication: 53
        |- copying: 21
        |- reduction: 32
    |- computation: 46
    |- layout-overhead: 0


RANK 1
PROFILING RESULTS:
|- multiply: 134
    |- communication: 53
        |- copying: 26
        |- reduction: 27
    |- computation: 51
    |- layout-overhead: 0


RANK 2
PROFILING RESULTS:
|- multiply: 134
    |- communication: 56
        |- copying: 19
        |- reduction: 37
    |- computation: 53
    |- layout-overhead: 0


RANK 3
PROFILING RESULTS:
|- multiply: 134
    |- communication: 60
        |- copying: 25
        |- reduction: 35
    |- computation: 56
    |- layout-overhead: 0
```
All the time measurements are given in milliseconds.

### Requirements
CARMA algorithm uses:
  - `MPI`
  - `OpenMP`
  - `dgemm` that is provided either through `MKL` (Intel Parallel Studio XE), or through `openblas`.

### Authors
Marko Kabic 

### Mentors/Supervisors
Professor Dr. Joost VandeVondele \
Dr. Raffaele Solcà
