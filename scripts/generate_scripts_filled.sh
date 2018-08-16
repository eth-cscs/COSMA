experiment_time="00:10:00"

n_nodes=(4 8 15 29 57 114 228 456 )
n_tasks=(128 256 512 1024 2048 4096 8192 16384 )
p_range=(12 28 56 112 224 452 908 1820 )
p_rows=(3 4 7 8 14 4 4 35 )
p_cols=(4 7 8 14 16 113 227 52 )
strong_scaling_square=16384

weak_scaling_p0=(16329 23094 31622 43969 61644 87177 123288 174355 )
weak_scaling_p1=(12960 16329 20136 25085 31422 39590 49880 62845 )

strong_scaling_thin_mn=17408
strong_scaling_thin_k=3735552

weak_scaling_p0_mn=(2052 2586 3189 3973 4977 6270 7900 9954 )
weak_scaling_p0_k=(129864 206152 313444 486494 763426 1211833 1923708 3053705 )

weak_scaling_p1_mn=(1759 2052 2360 2733 3176 3704 4322 5041 )
weak_scaling_p1_k=(95416 129864 171698 230218 310854 422986 575660 783310 )

mem_limit=5000000000

DATE=`date '+%d-%m-%Y[%H:%M:%S]'`
mkdir $DATE
cd ./$DATE

n_rep=1

files=()

for node_idx in ${!n_nodes[@]}
do
    if [ $node_idx -le 4 ]
    then
        m_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} ${weak_scaling_p0_mn[node_idx]} ${weak_scaling_p1_mn[node_idx]})
        n_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} ${weak_scaling_p0_mn[node_idx]} ${weak_scaling_p1_mn[node_idx]})
        k_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} ${weak_scaling_p0_k[node_idx]} ${weak_scaling_p1_k[node_idx]})
    else
        m_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} $strong_scaling_thin_mn ${weak_scaling_p0_mn[node_idx]} ${weak_scaling_p1_mn[node_idx]})
        n_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} $strong_scaling_thin_mn ${weak_scaling_p0_mn[node_idx]} ${weak_scaling_p1_mn[node_idx]})
        k_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} $strong_scaling_thin_k ${weak_scaling_p0_k[node_idx]} ${weak_scaling_p1_k[node_idx]})
    fi

    nodes=${n_nodes[node_idx]}
    tasks=${n_tasks[node_idx]}
    sname=script_$nodes.sh
    sed "s|GLOBAL_NODES|$nodes|g; s|GLOBAL_P|${p_rows[node_idx]}|g; s|GLOBAL_Q|${p_cols[node_idx]}|g; s|GLOBAL_TIME|$experiment_time|g; s|GLOBAL_MEM_LIMIT|$mem_limit|g; s|GLOBAL_M_RANGE|(${m_values[*]})|g; s|GLOBAL_N_RANGE|(${n_values[*]})|g; s|GLOBAL_K_RANGE|(${k_values[*]})|g; s|GLOBAL_TASKS|${tasks}|g;" ../mm_comparison.sh > $sname

    chmod a+x $sname
    files+=($sname)
done

echo "Generated the following files: "
echo ${files[*]}

for rep in `seq 1 1 $((n_rep))`
do
    for file in ${files[@]}
    do
        sbatch ./$file
        echo "Executing the script "$file
    done
done


