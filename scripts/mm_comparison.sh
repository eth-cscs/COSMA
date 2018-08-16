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
#weird_nodes=(4 7 8 13 16 25 27 32 37 61 64 81 93 128 201 216 256 333 473 512 )
#weird_ranks=(109 217 344 730 1332)
n_tasks_weird=(12 24 28 48 60 96 104 124 144 240 252 320 368 508 800 860 1020 1328 1888 2044)
p_weird=(3 4 4 6 6 8 8 4 12 15 14 16 16 4 25 20 30 16 32 28)
q_weird=(4 6 7 8 10 12 13 31 12 16 18 20 23 127 32 43 34 83 59 73)

m_range=GLOBAL_M_RANGE
n_range=GLOBAL_N_RANGE
k_range=GLOBAL_K_RANGE

n_tasks_upper=GLOBAL_TASKS

export n_iter=1

mem_limit=GLOBAL_MEM_LIMIT  #8847360 # in # of doubles and not in bytes

run_scalapack() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    p_rows=$5
    p_cols=$6
    idx=$7
    n_ranks_per_node=4
    n_threads_per_rank=9
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    n_ranks=$((p_rows*p_cols))

    if [ $k -gt $m ]; then
        srun -N $nodes -n $n_ranks -c $n_threads_per_rank --hint=nomultithread \
             $prefix/DLA-interface/build/miniapp/matrix_multiplication \
             -m $m -n $n -k $k --scalapack \
             -p $p_rows -q $p_cols \
             -r $n_iter --transb
    else
        srun -N $nodes -n $n_ranks -c $n_threads_per_rank --hint=nomultithread \
             $prefix/DLA-interface/build/miniapp/matrix_multiplication \
             -m $m -n $n -k $k --scalapack \
             -p $p_rows -q $p_cols \
             -r $n_iter
    fi

#    if [ $idx -eq 2 ]; then
#        echo ""
#        echo "============================"
#        echo "      PARTIAL NODE"
#        echo "============================"
#        n_ranks=$((4*(nodes-1)))
#        index=$(find_index $n_ranks)
#        p_rows=${p_weird[$index]}
#        p_cols=${q_weird[$index]}
#
#        if [ $k -gt $m ]; then
#            srun -N $nodes -n $n_ranks -c $n_threads_per_rank --hint=nomultithread \
#                 $prefix/DLA-interface/build/miniapp/matrix_multiplication \
#                 -m $m -n $n -k $k --scalapack \
#                 -p $p_rows -q $p_cols \
#                 -r $n_iter --transb
#        else
#            srun -N $nodes -n $n_ranks -c $n_threads_per_rank --hint=nomultithread \
#                 $prefix/DLA-interface/build/miniapp/matrix_multiplication \
#                 -m $m -n $n -k $k --scalapack \
#                 -p $p_rows -q $p_cols \
#                 -r $n_iter
#        fi
#    fi
}

run_carma() {
    m=$1
    n=$2
    k=$3
    nodes=$4
    limited=$5
    idx=$6
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    #n_ranks=$((nodes*n_ranks_per_node))
    n_ranks=$n_tasks_upper
    mem_limit=$((mem_limit/n_ranks_per_node))
    if [ $nodes -ge 100 ];
    then
        #mem_limit=$(echo "($mem_limit*0.4 + 0.5)/1"|bc)
        mem_limit=45000000
        echo "Decreased memory limit to "$mem_limit
    elif [ $nodes -ge 80 ];
    then
        mem_limit=$(echo "($mem_limit*0.5 + 0.5)/1"|bc)
    fi

    if [ "$limited" = true ]; 
    then
        srun -N $nodes -n $n_ranks \
             $prefix/CARMA/build/miniapp/temp-miniapp \
             -m $m -n $n -k $k -P $n_ranks --memory $mem_limit

        #if [ $idx -eq 2 ]; then
        #    echo ""
        #    echo "============================"
        #    echo "      PARTIAL NODE"
        #    echo "============================"
        #    n_ranks=$((36*(nodes-1)+1))
        #    echo "Total number of cores: "$n_ranks

        #    srun -N $nodes -n $n_ranks \
        #         $prefix/CARMA/build/miniapp/temp-miniapp \
        #         -m $m -n $n -k $k -P $n_ranks --memory $mem_limit
        #fi
    else
        srun -N $nodes -n $n_ranks \
             $prefix/CARMA/build/miniapp/temp-miniapp \
             -m $m -n $n -k $k -P $n_ranks

        #if [ $idx -eq 2 ]; then
        #    echo ""
        #    echo "============================"
        #    echo "      PARTIAL NODE"
        #    echo "============================"
        #    n_ranks=$((36*(nodes-1)+1))
        #    echo "Total number of cores: "$n_ranks

        #    srun -N $nodes -n $n_ranks \
        #         $prefix/CARMA/build/miniapp/temp-miniapp \
        #         -m $m -n $n -k $k -P $n_ranks
        #fi
    fi

    if [ $? -ne 0 ];
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
    idx=$6
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    #n_ranks=$((nodes*n_ranks_per_node))
    n_ranks=$n_tasks_upper
    mem_limit=$((mem_limit/n_ranks_per_node))

    if [ $nodes -ge 100 ];
    then
        #mem_limit=$(echo "($mem_limit*0.4 + 0.5)/1"|bc)
        mem_limit=45000000
        echo "Decreased memory limit to "$mem_limit
    elif [ $nodes -ge 80 ];
    then
        mem_limit=$(echo "($mem_limit*0.5 + 0.5)/1"|bc)
    fi

    if [ "$limited" = true ]; 
    then
        srun -N $nodes -n $n_ranks \
             $prefix/CAPS/rect-class/bench-rect-nc \
             -m $m -n $n -k $k -L $mem_limit
    else
        srun -N $nodes -n $n_ranks \
             $prefix/CAPS/rect-class/bench-rect-nc \
             -m $m -n $n -k $k 
    fi

    if [ $? -ne 0 ]; 
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
    idx=$6
    #n_ranks_per_node=36
    n_ranks_per_node=36
    n_threads_per_rank=1
    export OMP_NUM_THREADS=$n_threads_per_rank
    export MKL_NUM_THREADS=$n_threads_per_rank
    #n_ranks=$((nodes*n_ranks_per_node))
    n_ranks=$n_tasks_upper

    memory_in_bytes=$((8*mem_limit))
    memory_in_bytes=$(echo "($memory_in_bytes+0.5)/1"|bc)
    echo "Memory limit = "$memory_in_bytes

    if [ "$limited" = true ]; 
    then
        CTF_MEMORY_SIZE=$memory_in_bytes srun -N $nodes -n $n_ranks \
           $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
           -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
           -bench 1 -test 0 | grep -v -i ERROR && echo "error"

        #if [ $idx -eq 2 ]; then
        #    echo ""
        #    echo "============================"
        #    echo "      PARTIAL NODE"
        #    echo "============================"
        #    n_ranks=$((36*(nodes-1)+1))
        #    echo "Total number of cores: "$n_ranks

        #    CTF_MEMORY_SIZE=$memory_in_bytes srun -N $nodes -n $n_ranks \
        #       $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
        #       -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
        #       -bench 1 -test 0 | grep -v -i ERROR && echo "error"
        #fi
    else
        srun -N $nodes -n $n_ranks \
           $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
           -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
           -bench 1 -test 0 | grep -v -i ERROR && echo "error"

        #if [ $idx -eq 2 ]; then
        #    echo ""
        #    echo "============================"
        #    echo "      PARTIAL NODE"
        #    echo "============================"
        #    n_ranks=$((36*(nodes-1)+1))
        #    echo "Total number of cores: "$n_ranks

        #    srun -N $nodes -n $n_ranks \
        #       $prefix/ctf/build/bin/matmul -m $m -n $n -k $k \
        #       -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -niter $n_iter \
        #       -bench 1 -test 0 | grep -v -i ERROR && echo "error"
        #fi
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

function find_index() {
    local value=$1
    for i in "${!n_tasks_weird[@]}";
    do
        if [ "${n_tasks_weird[$i]}" -eq "$value" ]; then
            echo $i
            return 0
        fi
    done
    echo "-1"
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
    idx=$8

    echo ""
    echo ""
    echo "======================================================================================"
    echo "           EXPERIMENT: nodes = $nodes, (m, n, k) = ($m, $n, $k)"
    echo "======================================================================================"
    echo "memory limit = $mem_limit"
    echo $nodes" "$m" "$n" "$k" "$mem_limit >> "config.txt"

    if [ "$algorithm" = "carma" ];
    then
        echo ""
        echo "================================="
        echo "           CARMA"
        echo "================================="
        # OUR ALGORITHM
        output=$(run_carma $m $n $k $nodes true $idx)
        error=$(substring $output "error")
        echo "error = "$error
        if [ "$error" = "y" ];
        then 
            echo "Failed with limited memory, retrying with infinite memory..."
            output=$(run_carma $m $n $k $nodes false $idx)
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

    elif [ "$algorithm" = "scalapack" ];
    then
        echo ""
        echo "================================="
        echo "           SCALAPACK"
        echo "================================="

        # SCALAPACK
        output=$(run_scalapack $m $n $k $nodes $p_rows $p_cols $idx)
        scalapack_time=$(echo $output | awk -v n_iters="$n_iter" '/ScaLAPACK TIMES/ {for (i = 0; i < n_iters; i++) {printf "%d ", $(5+i)}}')
        echo $output
        echo "SCALAPACK TIME = "$scalapack_time
        time=`date '+[%H:%M:%S]'`
        echo $nodes" "$m" "$n" "$k" inf "$scalapack_time >> "scalapack_"$nodes".txt"
        echo "Finished ScaLAPACK algorithm at "$time

    elif [ "$algorithm" = "cyclops" ];
    then
        echo ""
        echo "================================="
        echo "           CYCLOPS"
        echo "================================="

        # CYCLOPS
        output=$(run_cyclops $m $n $k $nodes true $idx)
        error=$(substring $output "error")
        echo "error = "$error
        if [ "$error" = "y" ];
        then 
            echo "Failed with limited memory, retrying with infinite memory..."
            output=$(run_cyclops $m $n $k $nodes false $idx)
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

    elif [ "$algorithm" = "old_carma" ];
    then
        echo ""
        echo "================================="
        echo "           OLD CARMA"
        echo "================================="
        # OLD CARMA
        #if [ $(contains "${n_nodes_powers[@]}" $nodes) == "y" ]; then
        output=$(run_old_carma $m $n $k $nodes true $idx)
        error=$(substring $output "error")
        echo "error = "$error
        if [ "$error" = "y" ];
        then 
            echo "Failed with limited memory, retrying with infinite memory..."
            output=$(run_old_carma $m $n $k $nodes false $idx)
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
        #else
        #    echo "OLD CARMA TIME = not a power of 2"
        #    echo "not a power of 2" >> "old_carma_"$nodes".txt"
        #fi
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
    idx=$7

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
    output=$(run_carma $m $n $k $nodes true $idx)
    error=$(substring $output "error")
    echo "error = "$error
    if [ "$error" = "y" ];
    then 
        echo "Failed with limited memory, retrying with infinite memory..."
        output=$(run_carma $m $n $k $nodes false $idx)
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
    output=$(run_scalapack $m $n $k $nodes $p_rows $p_cols $idx)
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
    output=$(run_cyclops $m $n $k $nodes true $idx)
    error=$(substring $output "error")
    echo "error = "$error
    if [ "$error" = "y" ];
    then 
        echo "Failed with limited memory, retrying with infinite memory..."
        output=$(run_cyclops $m $n $k $nodes false $idx)
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
    #if [ $(contains "${n_nodes_powers[@]}" $nodes) == "y" ]; then
    output=$(run_old_carma $m $n $k $nodes true $idx)
    error=$(substring $output "error")
    echo "error = "$error
    if [ "$error" = "y" ];
    then 
        echo "Failed with limited memory, retrying with infinite memory..."
        output=$(run_old_carma $m $n $k $nodes false $idx)
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
    #else
    #    echo "OLD CARMA TIME = not a power of 2"
    #    echo "not a power of 2" >> "old_carma_"$nodes".txt"
    #fi
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
    #run_all $m $n $k GLOBAL_NODES GLOBAL_P GLOBAL_Q $idx
    #run_one $m $n $k GLOBAL_NODES GLOBAL_P GLOBAL_Q carma $idx
    run_one $m $n $k GLOBAL_NODES GLOBAL_P GLOBAL_Q old_carma $idx
done
