# COSMA Attributions:

COSMA uses the following external projects:
- [grid2grid](https://github.com/kabicm/grid2grid): used for transforming between different grid-like data layouts (initial distributions of matrices over MPI ranks). Licenced under the [BSD-3-Clause Licence](https://github.com/kabicm/grid2grid/blob/master/LICENCE).
- [Tiled-MM](https://github.com/kabicm/Tiled-MM): used for performing `dgemm` calls with the GPU-backend. Licenced under the [BSD-3-Clause Licence](https://github.com/kabicm/Tiled-MM/blob/master/LICENCE).
- [semiprof](https://github.com/bcumming/semiprof): used for profiling the code. Licenced under the [BSD-3-Clause Licence](https://github.com/bcumming/semiprof/blob/master/LICENCE).
- [options](https://github.com/kabicm/options): used for parsing the command line options. Licenced under the [BSD-3-Clause Licence](https://github.com/kabicm/options/blob/master/LICENCE).
- [googletest](https://github.com/google/googletest): used for unit testing. Licenced under the [BSD-3-Clause Licence](https://github.com/google/googletest/blob/master/LICENSE).
- [gtest_mpi](https://github.com/AdhocMan/gtest_mpi): used as a plugin for googletest adding the MPI support. Licenced under the [BSD-3-Clause Licence](https://github.com/AdhocMan/gtest_mpi/blob/master/LICENSE).

These projects are added as submodules and can be found in the `libs` folder.
