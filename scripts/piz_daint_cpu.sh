# load the necessary modules
module load daint-mc
module swap PrgEnv-cray PrgEnv-gnu
module load CMake
module unload cray-libsci
module load intel # defines $MKLROOT

# Setup the compiler
#
export CC=`which cc`
export CXX=`which CC`

# Enable dynamic linking
#
export CRAYPE_LINK_TYPE=dynamic

# Enable threading
# 
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
