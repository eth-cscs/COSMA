## Installing COSTA

### Step 1 - cloning the repo

In order to build COSTA, the first step it to clone the git repo as follows:
```bash
git clone https://github.com/kabicm/COSTA costa && cd costa
mkdir build && cd build
```
### Step 2 - building \& installing

This step depends on whether you are a scalapack user.

a) **For general users:** in this case it is enough to build as follows:
```bash
cmake -DCMAKE_INSTALL_PREFIX=<installation dir>/costa ..
make -j
make install
```

b) **For SCALAPACK users:** if you want `pxgemr2d` and `pxtran(u)` wrappers to be provided by COSTA, then scalapack has to be specified when building COSTA:
```bash
cmake -DCOSTA_SCALAPACK=MKL -DCMAKE_INSTALL_PREFIX=<installation dir>/costa ..
make -j
make install
```
The value of `COSTA_SCALAPACK` can be: `MKL`, `CRAY_LIBSCI` or `CUSTOM`. If `CUSTOM` scalapack is used, then the environment variable `SCALAPACK_ROOT` has to be defined.

The library is now installed in `<installation dir>/costa` that is specified when building the library.

## COSTA Dependencies

COSTA is a CMake project and requires a recent CMake (version >=3.12).

External dependencies are:
- `MPI 3`: (required)
- `OpenMP`: (required)
- `SCALAPACK`: (optional)

## Using COSTA

There are two ways how you can use COSTA in your own project:

a) Using COSTA within another CMake project:

COSTA is CMake friendly and provides a `costaConfig.cmake` module for easy integration into 3rd-party CMake projects with:
```cmake
find_package(costa REQUIRED)
target_link_libraries(... costa::costa)
```
COSTA's dependencies are taken care of internally, nothing else needs to be linked. Make sure to set `CMAKE_INSTALL_PREFIX` to COSTA's installation directory when building.

b) Using COSTA in a general project:

If your project is not a CMake project, then your code has to be linked to COSTA, i.e. to be linked to:
```bash
-L<installation dir>/costa/lib64 -lcosta
```

For SCALAPACK users, it is slightly different:  
```bash
-L<installation dir>/costa/lib64 -lcosta_scalapack
```
For SCALAPACL users it is important that your code is linked to COSTA before it is linked to SCALAPACK, because both COSTA and SCALAPACK provide `pxgemr2d` and `pxtran(u)` routines and the one that is first in the linking line will be used by the linker.

