#!/bin/bash -l
#SBATCH --job-name=matmul
#SBATCH --time=GLOBAL_TIME
#SBATCH --nodes=GLOBAL_NODES
#SBATCH --constraint=mc
set -x

module load daint-mc
module swap PrgEnv-cray PrgEnv-gnu
module unload cray-libsci
module load intel
module load CMake
export CC=`which cc`
export CXX=`which CC`
export CRAYPE_LINK_TYPE=dynamic

export prefix="../.."

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/scratch/snx3000/kabicm/ctf/build/lib_shared"

n_nodes_powers=(4 8 16 32 64 128 256 512)

# memory strong scaling experiments are not memory limited
# p0 and p1 experiments in weak scaling are memory limited
mem_limited_experiments=(false true true false true true)

m_range=GLOBAL_M_RANGE
n_range=GLOBAL_N_RANGE
k_range=GLOBAL_K_RANGE

export n_iter=1

mem_limit=GLOBAL_MEM_LIMIT  #8847360 # in # of doubles and not in bytes

run_scalapack() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    p_rows=$5
    p_cols=$6
    n_ranks_per_node=4
    n_threads_per_rank=9
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    srun -N $nodes -n $n_ranks -c $n_threads_per_rank --hint=nomultithread --ntasks-per-node=$n_ranks_per_node \
         $prefix/DLA-interface/build/miniapp/matrix_multiplication \
         -m $m -n $n -k $k --scalapack \
         -p $p_rows -q $p_cols \
         -r $n_iter
}

run_carma() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    limited_memory=$5
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))
    mem_limit=$((mem_limit/n_ranks_per_node))

    if [ "$limited_memory" = true ]; then
        echo "Using limited memory = "$mem_limit
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CARMA/build/miniapp/temp-miniapp \
             -m $m -n $n -k $k -P $n_ranks --memory $mem_limit
    else
        echo "Using unlimited memory"
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CARMA/build/miniapp/temp-miniapp \
             -m $m -n $n -k $k -P $n_ranks
    fi
}

run_old_carma() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    limited_memory=$5
    n_ranks_per_node=32
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))
    mem_limit=$((mem_limit/n_ranks_per_node))

    if [ "$limited_memory" = true ]; then
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CAPS/rect-class/bench-rect-nc \
             -m $m -n $n -k $k -L $mem_limit
    else
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CAPS/rect-class/bench-rect-nc \
             -m $m -n $n -k $k
    fi
}

run_cyclops() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    limited_memory=$5
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    memory_in_bytes=$((mem_limit/n_ranks_per_node*8))

    if [ "$limited_memory" = true ]; then
        CTF_MEMORY_SIZE=$memory_in_bytes srun -N $nodes -n $n_ranks \
               --ntasks-per-node=$n_ranks_per_node \
               $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
               -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
               -bench 1 -test 0
    else
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
               $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
               -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
               -bench 1 -test 0

    fi
}

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

run_all() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    p_rows=$5
    p_cols=$6
    limited_memory=$7

    echo "Performing m = "$m", n = "$n", k = "$k
    echo $nodes" "$m" "$n" "$k" "$mem_limit >> "config.txt"
    # OUR ALGORITHM
    output=$(run_carma $m $n $k $nodes $limited_memory)
    carma_time=$(echo $output | awk -v n_iters="$n_iter" '/CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
    echo "CARMA TIMES = "$carma_time
    echo $carma_time >> "carma.txt"
    time=`date '+[%H:%M:%S]'`
    echo "Finished our algorithm at "$time

    # SCALAPACK
    output=$(run_scalapack $m $n $k $nodes $p_rows $p_cols $limited_memory)
    scalapack_time=$(echo $output | awk -v n_iters="$n_iter" '/ScaLAPACK TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
    echo "SCALAPACK TIME = "$scalapack_time
    time=`date '+[%H:%M:%S]'`
    echo $scalapack_time >> "scalapack.txt"
    echo "Finished ScaLAPACK algorithm at "$time

    # CYCLOPS
    output=$(run_cyclops $m $n $k $nodes $limited_memory)
    cyclops_time=$(echo $output | awk -v n_iters="$n_iter" '/CYCLOPS TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
    time=`date '+[%H:%M:%S]'`
    echo "CYCLOPS TIME = "$cyclops_time
    echo $cyclops_time >> "cyclops.txt"
    echo "Finished CYCLOPS algorithm at "$time

    # OLD CARMA
    if [ $(contains "${n_nodes_powers[@]}" $nodes) == "y" ]; then
        output=$(run_old_carma $m $n $k $nodes $limited_memory)
        old_carma_time=$(echo $output | awk -v n_iters="$n_iter" '/OLD_CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
        echo "OLD CARMA TIME = "$old_carma_time
        echo $old_carma_time >> "old_carma.txt"
    else
        echo "OLD CARMA TIME = not a power of 2"
        echo "not a power of 2" >> "old_carma.txt"
    fi
    time=`date '+[%H:%M:%S]'`
    echo "Finished OLD CARMA algorithm at "$time
}

compile() {
    # compile all libraries
    (
        echo "Compiling CAPS (OLD_CARMA) library..."
        cd $prefix/CAPS/rect-class/
        make -j
    )
    (
        echo "Compiling our library..."
        cd $prefix/CARMA/build/
        make -j
    )
    (
        echo "Compiling DLA-interface (ScaLAPACK)"
        cd $prefix/DLA-interface/build/
        make -j
    )
    (
        echo "Compiling Cyclops library"
        cd $prefix/ctf/build/
        make -j
        make matmul
    )
}

IFS=

compile

for idx in ${!m_range[@]}
do
    m=${m_range[idx]}
    n=${n_range[idx]}
    k=${k_range[idx]}
    echo "Performing: m = "$m", n = "$n", k = "$k", nodes = GLOBAL_NODES"
    run_all $m $n $k GLOBAL_NODES GLOBAL_P GLOBAL_Q ${mem_limited_experiments[idx]}
done
