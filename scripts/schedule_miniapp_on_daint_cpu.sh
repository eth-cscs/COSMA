#!/bin/bash -l
#SBATCH --job-name=cosma_miniapp
#SBATCH --constraint=mc
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=2
#SBATCH --time=2
#SBATCH --output=cosma_miniapp.out
#SBATCH --error=cosma_miniapp.err

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

# Move to `build` directory if not there already
#
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
# COSMA_DIR="$( dirname $SCRIPT_DIR )"
COSMA_DIR=$SCRATCH/cosma-master
MINIAPP_PATH=${COSMA_DIR}/build/miniapp

# Run tests
#
echo "====================="
echo "   Square Matrices"
echo "====================="
echo "(m, n, k) = (10000, 10000, 10000)"
echo "Nodes: 10"
echo "MPI processes per rank: 2"
echo ""
srun ${MINIAPP_PATH}/scalars_miniapp -m 10000 -n 10000 -k 10000 -P 20

echo ""
echo "====================="
echo "    Tall Matrices"
echo "====================="
echo "(m, n, k) = (1000, 1000, 1000000)"
echo "Nodes: 10"
echo "MPI processes per rank: 2"
echo ""
srun ${MINIAPP_PATH}/scalars_miniapp -m 1000 -n 1000 -k 1000000 -P 20


