experiment_time="00:15:00"

n_nodes=(4 7 8 13 16 25 27 32 37 61 64 81 93 128 201 216 256 333 473 512 )
p_range=(16 28 32 52 64 100 108 128 148 244 256 324 372 512 804 864 1024 1332 1892 2048 )
p_rows=(4 4 4 4 8 10 9 8 4 4 16 18 12 16 12 24 32 36 43 32 )
p_cols=(4 7 8 13 8 10 12 16 37 61 16 18 31 32 67 36 32 37 44 64 )
strong_scaling_square=16384

weak_scaling_p0=(16329 21602 23094 29439 32659 40824 42426 46188 49665 63770 65319 73484 78740 92376 115758 120000 130639 148996 177576 184752 )
weak_scaling_p1=(12960 15618 16329 19198 20574 23874 24494 25922 27207 32141 32659 35327 36992 41148 47828 48989 51844 56593 63617 65319 )

strong_scaling_thin_mn=17408
strong_scaling_thin_k=3735552

weak_scaling_p0_mn=(2052 2474 2586 3040 3258 3781 3879 4106 4309 5090 5173 5595 5859 6517 7575 7759 8212 8964 10076 10346 )
weak_scaling_p0_k=(129864 188623 206152 284952 327274 440705 463842 519558 572362 798672 824736 964917 1057970 1309098 1768606 1855562 2078236 2476285 3129264 3298948 )

weak_scaling_p1_mn=(1759 1992 2052 2286 2394 2644 2690 2793 2885 3224 3258 3434 3540 3801 4202 4270 4434 4701 5082 5173 )
weak_scaling_p1_k=(95416 122383 129864 161158 176740 215502 222998 240505 256536 320322 327274 363410 386366 445378 544241 561997 605991 681221 796164 824736 )

mem_limit=5000000000

DATE=`date '+%d-%m-%Y[%H:%M:%S]'`
mkdir $DATE
cd ./$DATE

n_rep=1

files=()

for node_idx in ${!n_nodes[@]}
do
    if [ $node_idx -le 10 ]
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
    sname=script_$nodes.sh
    sed "s|GLOBAL_NODES|$nodes|g; s|GLOBAL_P|${p_rows[node_idx]}|g; s|GLOBAL_Q|${p_cols[node_idx]}|g; s|GLOBAL_TIME|$experiment_time|g; s|GLOBAL_MEM_LIMIT|$mem_limit|g; s|GLOBAL_M_RANGE|(${m_values[*]})|g; s|GLOBAL_N_RANGE|(${n_values[*]})|g; s|GLOBAL_K_RANGE|(${k_values[*]})|g" ../mm_comparison.sh > $sname

    chmod a+x $sname
    files+=($sname)
done

echo "Generated the following files: "
echo ${files[*]}

for rep in `seq 1 1 $((n_rep))`
do
    for file in ${files[@]}
    do
        #sbatch ./$file
        echo "Executing the script "$file
    done
done


