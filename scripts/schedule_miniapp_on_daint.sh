#!/bin/bash -l
#SBATCH --job-name=cosma_tests
#SBATCH --constraint=mc
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=2
#SBATCH --time=1
#SBATCH --output=cosma_tests.out
#SBATCH --error=cosma_tests.err

module load daint-mc
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
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
COSMA_DIR="$( dirname $SCRIPT_DIR )"
MINIAPP_PATH=${COSMA_DIR}/build/miniapp

# Run tests
#
echo "Tall Skinny Matrices\n"
srun ${MINIAPP_PATH}/scalars_miniapp -m 17408 -n 17408 -k 3735552 -P 128 -s pm64,pm2

echo "Square Matrices\n"
srun ${MINIAPP_PATH}/scalars_miniapp -m 16384 -n 16384 -k 16384 -P 128 -s pn8,pk4,pm4


