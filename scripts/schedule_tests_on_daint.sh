#!/bin/bash -l
#SBATCH --job-name=cosma_tests
#SBATCH --constraint=mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
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

# Move to `build` directory if not there already
#
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
COSMA_DIR="$( dirname $SCRIPT_DIR )"
mkdir -p ${COSMA_DIR}/build
cd ${COSMA_DIR}/build

# Build tests if not already built
#
make tests

# Run tests
#
srun tests/test.multiply
srun -n 1 tests/test.mapper
srun -n 1 tests/test.strategy
