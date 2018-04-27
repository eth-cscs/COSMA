waters=(64)
n_nodes=(16 32 48 64)
p_proc=(24 24 36 48)
q_proc=(24 48 48 48)
ranks_per_node=36

run_carma() {
    w=$1
    nodes=$2

    m=$((w*136))
    n=$m
    k=$((w*w*228))
    ranks=$((ranks_per_node * nodes))

    srun -N $nodes -n $ranks ./build/miniapp/temp-miniapp -m $m -n $n -k $k -P $ranks
}

run_scalapack() {
    w=$1
    nodes=$2
    p=$3
    q=$4

    m=$((w*136))
    n=$m
    k=$((w*w*228))
    ranks=$((ranks_per_node * nodes))

    srun -N $nodes -n $ranks ../DLA-interface/build/test/extra/matrix_multiplication -m $m -n $n -k $k -p $p -q $q --scalapack
}

IFS=
for water in ${waters[@]}
do 
    for proc_id in ${!n_nodes[@]}
    do
        p=${p_proc[proc_id]}
        q=${q_proc[proc_id]}
        n_nodes=${n_nodes[proc_id]}

        output=$(run_carma $water $n_nodes)
        carma_time=$(echo $output | awk '/CARMA AVG TIME/ {print $5}')
        echo "CARMA TIME = "$carma_time
        echo $carma_time >> "carma.txt"

        output_scalapack=$(run_scalapack $water $n_nodes $p $q)
        scalapack_time=$(echo $output_scalapack | awk '/Scalapack AVG TIME/ {print $5}')
        echo "SCALAPACK TIME = "$scalapack_time
        echo $scalapack_time >> "scalapack.txt"
    done
done




