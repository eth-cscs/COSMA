# load the necessary modules
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load CMake
module unload cray-libsci
module load intel # defines $MKLROOT
module load cudatoolkit

# Setup the compiler
#
export CC=`which cc`
export CXX=`which CC`

# Enable dynamic linking
#
export CRAYPE_LINK_TYPE=dynamic
export CRAY_CUDA_MPS=1

# Enable threading
# 
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
