n_nodes_list=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36)

run() {

    n_nodes=$1
    srun -N $n_nodes -n $n_nodes ./tests/ubench/ubench-allgather
}

IFS=

for n_nodes in ${n_nodes_list[@]}
do 
    echo "NODES = "$n_nodes
    output=$(run $n_nodes)
    avg_time_v=$(echo $output | awk '/MPI_Allgatherv AVG TIME/ {print $5}')
    avg_time=$(echo $output | awk '/MPI_Allgather AVG TIME/ {print $5}')
    echo $output
    echo "avg_time_v = "$avg_time_v
    echo "avg_time = "$avg_time
    echo $avg_time_v >> "allgather_v.txt"
    echo $avg_time >> "allgather.txt"
done
