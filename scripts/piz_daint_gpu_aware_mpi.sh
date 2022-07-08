# load the necessary modules
module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module swap gcc/11.2.0 gcc/9.3.0
module unload cray-libsci
module load intel # defines $MKLROOT
module load cudatoolkit
module load CMake

# Setup the compiler
#
export CC=`which cc`
export CXX=`which CC`

export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_NO_GPU_DIRECT=1
# export NCCL_ROOT=/scratch/snx3000/kabicm/nccl/build
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
