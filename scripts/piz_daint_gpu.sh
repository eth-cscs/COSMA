# load the necessary modules
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module unload cray-libsci
module load intel # defines $MKLROOT
module load cudatoolkit
module load CMake

# Setup the compiler
#
export CC=`which cc`
export CXX=`which CC`

export NCCL_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/comm_libs/nccl
# export NCCL_PKG_CONFIG=/opt/nvidia/hpc_sdk/Linux_x86_64/21.3/comm_libs/nccl/lib/pkgconfig/
# export PKG_CONFIG_PATH=${NCCL_PKG_CONFIG}:${PKG_CONFIG_PATH}

# Enable dynamic linking
#
export CRAYPE_LINK_TYPE=dynamic
export CRAY_CUDA_MPS=1

# Enable threading
# 
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
