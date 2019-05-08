#!/bin/bash -l
#SBATCH --job-name=matmul
#SBATCH --time=30
#SBATCH --nodes=10
#SBATCH --constraint=mc
#SBATCH --output=cosma_breakdown.out
#set -x

module load daint-mc
module swap PrgEnv-cray PrgEnv-gnu
module unload cray-libsci
module load intel
module load CMake
module load hwloc

export CC=`which cc`
export CXX=`which CC`
export CRAYPE_LINK_TYPE=dynamic
export prefix="/scratch/snx3000/kabicm"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/scratch/snx3000/kabicm/ctf/build/lib_shared"

compile() {
    # compile all libraries
    echo "======================================================================================"
    echo "           COMPILATION"
    echo "======================================================================================"
    echo ""
    (
        echo "Compiling our library..."
        cd $prefix/cosma/build/
        make -j
    )
}

IFS=

compile

export n_iter=2
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export n_ranks=20

m_range=(10000 20000 40000 50000 60000 80000 100000)
n_range=(1000 2000 4000 6000 8000 10000 20000)
k_range=(10000 20000 40000 50000 60000 80000 100000)

for m in ${m_range[@]}
do
    for n in ${n_range[@]}
    do
        for k in ${k_range[@]}
        do
            mult=$((m*n*k))
            if [ $mult -gt 1000000000000 ]; then
                echo "======================================================================================"
                echo "           EXPERIMENT: nodes = $nodes, (m, n, k) = ($m, $n, $k)"
                echo "======================================================================================"

                echo "==================================="
                echo "           WITHOUT OVERLAP" 
                echo "==================================="

                srun -N 10 -n $n_ranks -p -y $prefix/cosma/build/miniapp/cosma-temp \
                    -m $m -n $n -k $k -P $n_ranks

                echo "==================================="
                echo "             WITH OVERLAP"
                echo "==================================="

                srun -N 10 -n $n_ranks -p -y $prefix/cosma/build/miniapp/cosma-temp \
                    -m $m -n $n -k $k -P $n_ranks -s bm2,bk5,bm2 --overlap

                srun -N 10 -n $n_ranks -p -y $prefix/cosma/build/miniapp/cosma-temp \
                    -m $m -n $n -k $k -P $n_ranks -s bk5,bm4 --overlap

                srun -N 10 -n $n_ranks -p -y $prefix/cosma/build/miniapp/cosma-temp \
                    -m $m -n $n -k $k -P $n_ranks -s bn2,bk5,bm2 --overlap

                srun -N 10 -n $n_ranks -p -y $prefix/cosma/build/miniapp/cosma-temp \
                    -m $m -n $n -k $k -P $n_ranks -s bk2,bn2,bm5 --overlap

                srun -N 10 -n $n_ranks -p -y $prefix/cosma/build/miniapp/cosma-temp \
                    -m $m -n $n -k $k -P $n_ranks -s bn2,bk2,bm5 --overlap

                time=`date '+[%H:%M:%S]'`
                echo "Finished our algorithm at "$time
            fi
        done
    done
done
