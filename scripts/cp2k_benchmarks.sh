#!/bin/bash -l
#SBATCH --job-name=cosma_cp2k_bench
#SBATCH --constraint=mc
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=2
#SBATCH --time=20
#SBATCH --output=cosma_cp2k_bench.out
#SBATCH --error=cosma_cp2k_bench.err

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
COSMA_DIR=$SCRATCH/cosma
MINIAPP_PATH=${COSMA_DIR}/build/miniapp

# Run tests
#

# 64 waters
waters=64
M=$((136*waters)) # 8704
N=$M # 8704
K=$((212*waters*waters)) # 868352
block_small=68 # M/128 because there are 128 processors in a row
block_medium=2176
block_large=3392

echo "64 H2O benchmark\n"

srun -N 128 -n 256 -C mc ${MINIAPP_PATH}/pdgemm-miniapp -m $M -n $N -k $K -P 256 \
    --block_a "${block_medium}x${block_large}" \
    --block_b "${block_large}x${block_medium}" \
    --block_c "${block_small}x${block_small}" \
    -p 128 -q 2 --scalapack

srun -N 128 -n 256 -C mc ${MINIAPP_PATH}/pdgemm-miniapp -m $M -n $N -k $K -P 256 \
    --block_a "${block_medium}x${block_large}" \
    --block_b "${block_large}x${block_medium}" \
    --block_c "${block_small}x${block_small}" \
    -p 128 -q 2 --cosma

# 128 waters
waters=128
M=$((136*waters)) # 17408
N=$M # 17408
K=$((212*waters*waters)) # 3473408
block_small=136 # M/128, because there are 128 processors in a row
block_medium=4352
block_large=13568

echo "128 H2O benchmark\n"

srun -N 128 -n 256 -C mc ${MINIAPP_PATH}/pdgemm-miniapp -m $M -n $N -k $K -P 256 \
    --block_a "${block_medium}x${block_large}" \
    --block_b "${block_large}x${block_medium}" \
    --block_c "${block_small}x${block_small}" \
    -p 128 -q 2 --scalapack

srun -N 128 -n 256 -C mc ${MINIAPP_PATH}/pdgemm-miniapp -m $M -n $N -k $K -P 256 \
    --block_a "${block_medium}x${block_large}" \
    --block_b "${block_large}x${block_medium}" \
    --block_c "${block_small}x${block_small}" \
    -p 128 -q 2 --cosma

