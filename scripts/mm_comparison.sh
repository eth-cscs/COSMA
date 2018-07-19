prefix="../"

m_range=(2048)
n_range=(2048)
k_range=(2048)

p_range=(16)
p_rows=(4)
p_cols=(4)

n_nodes=(1)

# scalapack configuration
block_size=128
ranks_per_node=16  # determined by flat --ntasks-per-node
threads_per_rank=1 # determined by flag -c

n_iter=5

mem_limit=1000000000 # in # of doubles and not in bytes

run_scalapack() {
    m=$1
    n=$2
    k=$3
    proc_id=$4
    n_ranks=${p_range[proc_id]}
    n_nodes=$((n_ranks/ranks_per_node))

    srun -N $n_nodes -n $n_ranks --ntasks-per-node $ranks_per_node \
         -c $threads_per_rank ../DLA-interface/build/miniapp/matrix_multiplication \
         -m $m -n $n -k $k --scalapack \
         -p ${p_rows[proc_id]} -q ${p_cols[proc_id]} \
         -r $n_iter
}

run_carma() {
    m=$1
    n=$2
    k=$3
    proc_id=$4
    n_ranks=${p_range[proc_id]}
    n_nodes=$((n_ranks/ranks_per_node))

    srun -N $n_nodes -n $n_ranks --ntasks-per-node $ranks_per_node \
         ../CARMA/build/miniapp/temp-miniapp \
         -m $m -n $n -k $k -P $n_ranks --memory=$mem_limit
}

run_old_carma() {
    m=$1
    n=$2
    k=$3
    proc_id=$4
    n_ranks=${p_range[proc_id]}
    n_nodes=$((n_ranks/ranks_per_node))

    srun -N $n_nodes -n $n_ranks --ntasks-per-node $ranks_per_node \
         ../CAPS/rect-class/bench-rect-nc \
         -m $m -n $n -k $k -L=$mem_limit
}

run_cyclops() {
    m=$1
    n=$2
    k=$3
    proc_id=$4
    n_ranks=${p_range[proc_id]}
    n_nodes=$((n_ranks/ranks_per_node))

    memory_in_bytes=$((mem_limit*8))

    CTF_PPN=$ranks_per_node CTF_MEMORY_SIZE=$memory_in_bytes srun -N $n_nodes \
           -n $n_ranks --ntasks-per-node $ranks_per_node \
           ../ctf/build/bin/matmul -m $m -n $n -k $k \
           -sym_A NS -sym_B NS -sym_C NS -sp_A 1.0 -sp_B 1.0 -sp_C 1.0 -n_iter $n_iter \
           -bench 1 -test 0
}

IFS=

# compile all libraries
echo "Compiling CAPS (OLD_CARMA) library..."
cd ../CAPS/rect-class/
make -j
echo "Compiling our library..."
cd ../../CARMA/build/
make -j
echo "Compiling DLA-interface (ScaLAPACK)"
cd ../../DLA-interface/build/
make -j
echo "Compiling Cyclops library"
cd ../../ctf/build/
make -j
make matmul
cd ../../benchmarks


for m in ${m_range[@]}
do
    for n in ${n_range[@]}
    do
        for k in ${k_range[@]}
        do
            for proc_id in ${!p_range[@]}
            do
                echo "Performing m = "$m", n = "$n", k = "$k
                # OUR ALGORITHM
                output=$(run_carma $m $n $k $proc_id)
                carma_time=$(echo $output | awk '/CARMA MIN TIME/ {print $6}')
                echo "CARMA TIME = "$carma_time
                echo $carma_time >> "carma.txt"

                # SCALAPACK
                output=$(run_scalapack $m $n $k $proc_id)
                scalapack_time=$(echo $output | awk '/ScaLAPACK: Best/ {print 1000*$3}')
                echo "SCALAPACK TIME = "$scalapack_time
                echo $scalapack_time >> "scalapack.txt"

                # OLD CARMA
                output=$(run_old_carma $m $n $k $proc_id)
                old_carma_time=$(echo $output | awk '/OLD_CARMA MIN TIME/ {print $5}')
                echo "OLD CARMA TIME = "$old_carma_time
                echo $old_carma_time >> "old_carma.txt"

                # CYCLOPS
                output=$(run_cyclops $m $n $k $proc_id)
                cyclops_time=$(echo $output | awk '/CYCLOPS MIN TIME/ {print $5}')
                echo "CYCLOPS TIME = "$cyclops_time
                echo $cyclops_time >> "cyclops.txt"
            done
        done
    done
done
