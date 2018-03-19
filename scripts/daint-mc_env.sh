module switch PrgEnv-cray PrgEnv-gnu
module load daint-mc
module load CMake
module load intel           # defines $MKLROOT

# enable the dynamic linking and
# the asynchronous thread progressing
# MPICH (on Cray systems)
export CRAYPE_LINK_TYPE=dynamic
export MPICH_NEMESIS_ASYNC_PROGRESS=MC
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_GNI_ASYNC_PROGRESS_TIMEOUT=0

# setup the right compilers
export CC=`which gcc`
export CXX=`which g++`
