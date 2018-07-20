#!/bin/bash -l
#SBATCH --job-name=matmul
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --constraint=mc

module load daint-mc
module swap PrgEnv-cray PrgEnv-gnu
module load intel
module load CMake
export CC=`which cc`
export CXX=`which CC`
export CRAYPE_LINK_TYPE=dynamic

DATE=`date '+%d-%m-%Y[%H:%M:%S]'`
mkdir $DATE
cd ./$DATE
prefix="../.."

n_nodes=(1)
m_range=(2048)
n_range=(2048)
k_range=(2048)

p_range=(4)
p_rows=(2)
p_cols=(2)

export n_iter=5

mem_limit=1000000000 # in # of doubles and not in bytes

run_scalapack() {
    m=$1
    n=$2
    k=$3
    nodes_id=$4
    nodes=${n_nodes[nodes_id]}
    n_ranks_per_node=4
    n_threads_per_rank=9
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
         $prefix/DLA-interface/build/miniapp/matrix_multiplication \
         -m $m -n $n -k $k --scalapack \
         -p ${p_rows[nodes_id]} -q ${p_cols[nodes_id]} \
         -r $n_iter
}

run_carma() {
    m=$1
    n=$2
    k=$3
    nodes_id=$4
    nodes=${n_nodes[nodes_id]}
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
         $prefix/CARMA/build/miniapp/temp-miniapp \
         -m $m -n $n -k $k -P $n_ranks --memory=$mem_limit
}

run_old_carma() {
    m=$1
    n=$2
    k=$3
    nodes_id=$4
    nodes=${n_nodes[nodes_id]}
    n_ranks_per_node=32
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
         $prefix/CAPS/rect-class/bench-rect-nc \
         -m $m -n $n -k $k -L=$mem_limit
}

run_cyclops() {
    m=$1
    n=$2
    k=$3
    nodes_id=$4
    nodes=${n_nodes[nodes_id]}
    n_ranks_per_node=4
    n_threads_per_rank=9
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    memory_in_bytes=$((mem_limit*8))

    CTF_PPN=$n_ranks_per_node CTF_MEMORY_SIZE=$memory_in_bytes srun -N $nodes \
           -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
           $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
           -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -n_iter $n_iter \
           -bench 1 -test 0
}

IFS=

# compile all libraries
echo "Compiling CAPS (OLD_CARMA) library..."
cd $prefix/CAPS/rect-class/
make -j
cd ../../benchmarks/$DATE
echo "Compiling our library..."
cd $prefix/CARMA/build/
make -j
cd ../../benchmarks/$DATE
echo "Compiling DLA-interface (ScaLAPACK)"
cd $prefix/DLA-interface/build/
make -j
cd ../../benchmarks/$DATE
echo "Compiling Cyclops library"
cd $prefix/ctf/build/
make -j
make matmul
cd ../../benchmarks/$DATE

for nodes_idx in ${!n_nodes[@]}
do
    for m in ${m_range[@]}
    do
        for n in ${n_range[@]}
        do
            for k in ${k_range[@]}
            do
                nodes=${n_nodes[nodes_idx]}
                echo "Performing m = "$m", n = "$n", k = "$k
                echo $nodes" "$m" "$n" "$k" "$mem_limit >> "config.txt"
                # OUR ALGORITHM
                output=$(run_carma $m $n $k $nodes_idx)
                carma_time=$(echo $output | awk -v n_iters="$n_iter" '/CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
                echo "CARMA TIMES = "$carma_time
                echo $carma_time >> "carma.txt"

                # SCALAPACK
                output=$(run_scalapack $m $n $k $nodes_idx)
                scalapack_time=$(echo $output | awk -v n_iters="$n_iter" '/ScaLAPACK TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
                echo "SCALAPACK TIME = "$scalapack_time
                echo $scalapack_time >> "scalapack.txt"

                # OLD CARMA
                output=$(run_old_carma $m $n $k $nodes_idx)
                old_carma_time=$(echo $output | awk -v n_iters="$n_iter" '/OLD_CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
                echo "OLD CARMA TIME = "$old_carma_time
                echo $old_carma_time >> "old_carma.txt"

                # CYCLOPS
                output=$(run_cyclops $m $n $k $nodes_idx)
                cyclops_time=$(echo $output | awk -v n_iters="$n_iter" '/CYCLOPS TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
                echo "CYCLOPS TIME = "$cyclops_time
                echo $cyclops_time >> "cyclops.txt"
            done
        done
    done
done
