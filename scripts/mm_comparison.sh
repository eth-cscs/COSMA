#!/bin/bash -l
#SBATCH --job-name=matmul
#SBATCH --time=GLOBAL_TIME
#SBATCH --nodes=GLOBAL_NODES
#SBATCH --constraint=mc
#set -x

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
    limited=$5
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))
    mem_limit=$((mem_limit/n_ranks_per_node))

    if [ "$limited" = true ]; 
    then
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CARMA/build/miniapp/temp-miniapp \
             -m $m -n $n -k $k -P $n_ranks --memory $mem_limit
    else
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CARMA/build/miniapp/temp-miniapp \
             -m $m -n $n -k $k -P $n_ranks
    fi

    if [ $? -ne 0 ] 
    then
        echo "error"
    fi
}

run_old_carma() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    limited=$5
    n_ranks_per_node=32
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))
    mem_limit=$((mem_limit/n_ranks_per_node))

    if [ "$limited" = true ]; 
    then
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CAPS/rect-class/bench-rect-nc \
             -m $m -n $n -k $k -L $mem_limit
    else
        srun -N $nodes -n $n_ranks --ntasks-per-node=$n_ranks_per_node \
             $prefix/CAPS/rect-class/bench-rect-nc \
             -m $m -n $n -k $k 
    fi

    if [ $? -ne 0 ] 
    then
        echo "error"
    fi
}

run_cyclops() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    limited=$5
    #n_ranks_per_node=36
    n_ranks_per_node=1
    n_threads_per_rank=36
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((nodes*n_ranks_per_node))

    memory_in_bytes=$((8*mem_limit/n_ranks_per_node))
    memory_in_bytes=$(echo "(2*$mem_limit+0.5)/1"|bc)
    echo "Memory limit = "$memory_in_bytes

    if [ "$limited" = true ]; 
    then
        CTF_MEMORY_SIZE=$memory_in_bytes CTF_PPN=$n_ranks_per_node srun -N $nodes -n $n_ranks \
           --ntasks-per-node=$n_ranks_per_node \
           $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
           -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
           -bench 1 -test 0 | grep -v -i ERROR && echo "error"
    else
        CTF_PPN=$n_ranks_per_node srun -N $nodes -n $n_ranks \
           --ntasks-per-node=$n_ranks_per_node \
           $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
           -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
           -bench 1 -test 0 | grep -v -i ERROR && echo "error"
    fi

    if [ $? -ne 0 ] 
    then
        echo "error"
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

substring() {
    string="$1"
    substring="$2"
    if [ "${string/$substring}" = "$string" ];
    then
        echo "n"
    else
        echo "y"
    fi
}

run_one() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    p_rows=$5
    p_cols=$6
    algorithm=$7

    echo ""
    echo ""
    echo "======================================================================================"
    echo "           EXPERIMENT: nodes = $nodes, (m, n, k) = ($m, $n, $k)"
    echo "======================================================================================"
    echo "memory limit = $mem_limit"
    echo $nodes" "$m" "$n" "$k" "$mem_limit >> "config.txt"

    if [ "$algorithm" = "carma" ]
    then
        echo ""
        echo "================================="
        echo "           CARMA"
        echo "================================="
        # OUR ALGORITHM
        output=$(run_carma $m $n $k $nodes true)
        error=$(substring $output "error")
        echo "error = "$error
        if [ "$error" = "y" ];
        then 
            echo "Failed with limited memory, retrying with infinite memory..."
            output=$(run_carma $m $n $k $nodes false)
        fi
        echo $output
        carma_time=$(echo $output | awk -v n_iters="$n_iter" '/CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
        echo "CARMA TIMES = "$carma_time
        if [ "$error" = "y" ];
        then
            echo $nodes" "$m" "$n" "$k" inf "$carma_time >> "carma_"$nodes".txt"
        else
            echo $nodes" "$m" "$n" "$k" "$mem_limit" "$carma_time >> "carma_"$nodes".txt"
        fi
        time=`date '+[%H:%M:%S]'`
        echo "Finished our algorithm at "$time

    elif [ "$algorithm" = "scalapack" ] 
    then
        echo ""
        echo "================================="
        echo "           SCALAPACK"
        echo "================================="

        # SCALAPACK
        output=$(run_scalapack $m $n $k $nodes $p_rows $p_cols)
        scalapack_time=$(echo $output | awk -v n_iters="$n_iter" '/ScaLAPACK TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
        echo $output
        echo "SCALAPACK TIME = "$scalapack_time
        time=`date '+[%H:%M:%S]'`
        echo $nodes" "$m" "$n" "$k" inf "$scalapack_time >> "scalapack_"$nodes".txt"
        echo "Finished ScaLAPACK algorithm at "$time

    elif [ "$algorithm" = "cyclops" ]
    then
        echo ""
        echo "================================="
        echo "           CYCLOPS"
        echo "================================="

        # CYCLOPS
        output=$(run_cyclops $m $n $k $nodes true)
        error=$(substring $output "error")
        echo "error = "$error
        if [ "$error" = "y" ];
        then 
            echo "Failed with limited memory, retrying with infinite memory..."
            output=$(run_cyclops $m $n $k $nodes false)
        fi
        cyclops_time=$(echo $output | awk -v n_iters="$n_iter" '/CYCLOPS TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
        time=`date '+[%H:%M:%S]'`
        echo $output
        echo "CYCLOPS TIME = "$cyclops_time
        if [ "$error" = "y" ];
        then
            echo $nodes" "$m" "$n" "$k" inf "$cyclops_time >> "cyclops_"$nodes".txt"
        else
            echo $nodes" "$m" "$n" "$k" "$mem_limit" "$cyclops_time >> "cyclops_"$nodes".txt"
        fi
        echo "Finished CYCLOPS algorithm at "$time

    elif [ "$algorithm" = "old_carma" ]
    then
        echo ""
        echo "================================="
        echo "           OLD CARMA"
        echo "================================="
        # OLD CARMA
        if [ $(contains "${n_nodes_powers[@]}" $nodes) == "y" ]; then
            output=$(run_old_carma $m $n $k $nodes true)
            error=$(substring $output "error")
            echo "error = "$error
            if [ "$error" = "y" ];
            then 
                echo "Failed with limited memory, retrying with infinite memory..."
                output=$(run_old_carma $m $n $k $nodes false)
            fi
            old_carma_time=$(echo $output | awk -v n_iters="$n_iter" '/OLD_CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
            echo $output
            echo "OLD CARMA TIME = "$old_carma_time
            if [ "$error" = "y" ];
            then
                echo $nodes" "$m" "$n" "$k" inf "$old_carma_time >> "old_carma_"$nodes".txt"
            else
                echo $nodes" "$m" "$n" "$k" "$mem_limit" "$old_carma_time >> "old_carma_"$nodes".txt"
            fi
        else
            echo "OLD CARMA TIME = not a power of 2"
            echo "not a power of 2" >> "old_carma_"$nodes".txt"
        fi
        time=`date '+[%H:%M:%S]'`
        echo "Finished OLD CARMA algorithm at "$time
    fi
}

run_all() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    p_rows=$5
    p_cols=$6

    echo ""
    echo ""
    echo "======================================================================================"
    echo "           EXPERIMENT: nodes = $nodes, (m, n, k) = ($m, $n, $k)"
    echo "======================================================================================"
    echo "memory limit = $mem_limit"
    echo $nodes" "$m" "$n" "$k" "$mem_limit >> "config.txt"

    echo ""
    echo "================================="
    echo "           CARMA"
    echo "================================="
    # OUR ALGORITHM
    output=$(run_carma $m $n $k $nodes true)
    error=$(substring $output "error")
    echo "error = "$error
    if [ "$error" = "y" ];
    then 
        echo "Failed with limited memory, retrying with infinite memory..."
        output=$(run_carma $m $n $k $nodes false)
    fi
    echo $output
    carma_time=$(echo $output | awk -v n_iters="$n_iter" '/CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
    echo "CARMA TIMES = "$carma_time
    if [ "$error" = "y" ];
    then
        echo $nodes" "$m" "$n" "$k" inf "$carma_time >> "carma_"$nodes".txt"
    else
        echo $nodes" "$m" "$n" "$k" "$mem_limit" "$carma_time >> "carma_"$nodes".txt"
    fi
    time=`date '+[%H:%M:%S]'`
    echo "Finished our algorithm at "$time

    echo ""
    echo "================================="
    echo "           SCALAPACK"
    echo "================================="

    # SCALAPACK
    output=$(run_scalapack $m $n $k $nodes $p_rows $p_cols)
    scalapack_time=$(echo $output | awk -v n_iters="$n_iter" '/ScaLAPACK TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
    echo $output
    echo "SCALAPACK TIME = "$scalapack_time
    time=`date '+[%H:%M:%S]'`
    echo $nodes" "$m" "$n" "$k" inf "$scalapack_time >> "scalapack_"$nodes".txt"
    echo "Finished ScaLAPACK algorithm at "$time

    echo ""
    echo "================================="
    echo "           CYCLOPS"
    echo "================================="

    # CYCLOPS
    output=$(run_cyclops $m $n $k $nodes true)
    error=$(substring $output "error")
    echo "error = "$error
    if [ "$error" = "y" ];
    then 
        echo "Failed with limited memory, retrying with infinite memory..."
        output=$(run_cyclops $m $n $k $nodes false)
    fi
    cyclops_time=$(echo $output | awk -v n_iters="$n_iter" '/CYCLOPS TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
    time=`date '+[%H:%M:%S]'`
    echo $output
    echo "CYCLOPS TIME = "$cyclops_time
    if [ "$error" = "y" ];
    then
        echo $nodes" "$m" "$n" "$k" inf "$cyclops_time >> "cyclops_"$nodes".txt"
    else
        echo $nodes" "$m" "$n" "$k" "$mem_limit" "$cyclops_time >> "cyclops_"$nodes".txt"
    fi
    echo "Finished CYCLOPS algorithm at "$time

    echo ""
    echo "================================="
    echo "           OLD CARMA"
    echo "================================="
    # OLD CARMA
    if [ $(contains "${n_nodes_powers[@]}" $nodes) == "y" ]; then
        output=$(run_old_carma $m $n $k $nodes true)
        error=$(substring $output "error")
        echo "error = "$error
        if [ "$error" = "y" ];
        then 
            echo "Failed with limited memory, retrying with infinite memory..."
            output=$(run_old_carma $m $n $k $nodes false)
        fi
        old_carma_time=$(echo $output | awk -v n_iters="$n_iter" '/OLD_CARMA TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
        echo $output
        echo "OLD CARMA TIME = "$old_carma_time
        if [ "$error" = "y" ];
        then
            echo $nodes" "$m" "$n" "$k" inf "$old_carma_time >> "old_carma_"$nodes".txt"
        else
            echo $nodes" "$m" "$n" "$k" "$mem_limit" "$old_carma_time >> "old_carma_"$nodes".txt"
        fi
    else
        echo "OLD CARMA TIME = not a power of 2"
        echo "not a power of 2" >> "old_carma_"$nodes".txt"
    fi
    time=`date '+[%H:%M:%S]'`
    echo "Finished OLD CARMA algorithm at "$time

}

compile() {
    # compile all libraries
    echo "======================================================================================"
    echo "           COMPILATION"
    echo "======================================================================================"
    echo ""
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
    #run_all $m $n $k GLOBAL_NODES GLOBAL_P GLOBAL_Q
    run_one $m $n $k GLOBAL_NODES GLOBAL_P GLOBAL_Q cyclops
done
