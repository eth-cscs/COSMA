# COSMA Attributions:

COSMA uses the following external projects:
- [COSTA](https://github.com/eth-cscs/COSTA): used for transforming between COSMA and SCALAPACK matrix data layouts and for transposing distributed matrices. Licensed under the [BSD-3-Clause License](https://github.com/eth-cscs/COSTA/blob/master/LICENSE).
- [Tiled-MM](https://github.com/eth-cscs/Tiled-MM): used for performing `dgemm` calls with the GPU-backend. Licensed under the [BSD-3-Clause License](https://github.com/eth-cscs/Tiled-MM/blob/master/LICENSE).
- [semiprof](https://github.com/bcumming/semiprof): used for profiling the code. Licensed under the [BSD-3-Clause License](https://github.com/bcumming/semiprof/blob/master/LICENSE).
- [options](https://github.com/kabicm/options): used for parsing the command line options. Licensed under the [BSD-3-Clause License](https://github.com/kabicm/options/blob/master/LICENCE).
- [cxxopts](https://github.com/jarro2783/cxxopts): user for parsing the command line options. Licensed under the [MIT License](https://github.com/jarro2783/cxxopts/blob/master/LICENSE).
- [googletest](https://github.com/google/googletest): used for unit testing. Licensed under the [BSD-3-Clause License](https://github.com/google/googletest/blob/master/LICENSE).
- [gtest_mpi](https://github.com/AdhocMan/gtest_mpi): used as a plugin for googletest adding the MPI support. Licensed under the [BSD-3-Clause License](https://github.com/AdhocMan/gtest_mpi/blob/master/LICENSE).
- [interpose](https://github.com/ccurtsinger/interpose): used for dispatching some of the pxgemm calls to SCALAPACK. Licensed under the [MIT License](https://github.com/ccurtsinger/interpose/blob/master/COPYING.md).

Most of these projects are added as submodules and can be found in the `libs` folder.
