## Overview
CARMA (Demmel et al., 2013) is the first matrix-matrix multiplication algorithm that is communication-optimal for all memory ranges and all matrix shapes. [paper](http://www.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-205.pdf), [poster](http://www.cs.berkeley.edu/~odedsc/papers/CARMA%20Poster-SC12).

The algorithm recursively splits the largest matrix dimension creating smaller subproblems which are then recursively solved sequentially (DFS step) or in parallel (BFS step), depending on the available memory. When the base case of the recursion is reached, CARMA uses dgemm to perform the local multiplication. While appealing and simple at first sight, the implementation details are tricky and the distributed version requires the data layout very different from any layout used in existing linear-algebra libraries. Concretely, whenever the largest matrix is split, these two halves should already reside entirely on the corresponding halves of the processors. This data layout requirement applies recursively to the base case at which point any load balanced data layout can be used.

The original implementation of CARMA assumes that all the dimensions and the number of processors are the powers of 2 or that the number of processors always shares common divisors with the largest dimension in each step of the recursion. It also uses a cyclic data layout in the base case which requires that the data is locally completely reshuffled after each communication.

Here, we present the results from an implementation of CARMA that provides functionality not present in earlier published prototypes, namely the ability to deal with matrix dimensions and processor numbers that are not powers of two, and do not necessarily share common divisors. Furthermore, we derive a relatively simple underlying data layout, which preserves the communication-optimality of the algorithm, but requires less intermediate data reshufflings during execution, has improved memory access patterns and is potentially more compatible with existing linear algebra libraries.

## Using two remotes

```sh
git remote add both git@github.com:eth-cscs/CARMA.git
git remote set-url --add both git@gitlab.ethz.ch:kabicm/CARMA.git
```

## How to build for Piz Daint ?

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

CXX=CC CC=cc cmake -DCMAKE_CXX_FLAGS=-std=c++14 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBINDIR=$SCRATCH/carma/build \
  -DMAKE_TEST=on \
  -fopenmp \
  -DUSE_BLAS_DEFAULT=on \
  -DBLAS_INCLUDE_DIRS:PATH=$PROJECT/Env/daint-mc/include/ \
  ..

make -j 8
```

Instead of -DUSE_BLAS_DEFAULT, option -DUSE_BLAS_MKL can be used as well.

## How to test
In the build directory, do:
```
make test
```
### Requirements
You must have Intel Parallel Studio XE installed, which will provide MKL or use openblas instead.

### Authors
Marko Kabic \
Dr. Thibault Notargiacomo

### Mentors/Supervisors
Professor Dr. Joost VandeVondele \
Dr. Raffaele Solcà
