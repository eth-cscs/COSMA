experiment_time="00:30:00"

n_nodes=(4 7 8 13 16 25 27 32 37 61 64 81 93 128 201 216 256 333 453 473 512)

p_range=(16 28 32 52 64 100 108 128 148 244 256 324 372 512 804 864 1024 1332 1812 1892 2048)
p_rows=(4 4 4 4 8 10 9 8 4 4 16 18 12 16 12 24 32 36 12 43 32)
p_cols=(4 7 8 13 8 10 12 16 37 61 16 18 31 32 67 36 32 37 151 44 64)

strong_scaling_square=16384

weak_scaling_p0=(4206 5564 5948 7583 8413 10516 10928 11897 12793 16426 16826 18929 20283 23795 29818 30911 33652 38380 45742 47591 51519)
weak_scaling_p1=(3338 4023 4206 4945 5299 6149 6309 6677 7008 8279 8413 9100 9529 10599 12320 12619 13354 14578 16387 16826 17739)

strong_scaling_thin_mn=17408
strong_scaling_thin_k=3735552

weak_scaling_p0_mn=(262 347 371 473 525 657 683 743 799 1026 1051 1183 1267 1487 1863 1931 2103 2398 2858 2974 3219)
weak_scaling_p0_k=(67072 88832 94976 121088 134400 168192 174848 190208 204544 262656 269056 302848 324352 380672 476928 494336 538368 613888 731648 761344 824064)

weak_scaling_p1_mn=(208 251 262 309 331 384 394 417 438 517 525 568 595 662 770 788 834 911 1024 1051 1108)
weak_scaling_p1_k=(53248 64256 67072 79104 84736 98304 100864 106752 112128 132352 134400 145408 152320 169472 197120 201728 213504 233216 262144 269056 283648)

mem_limit=1000000000000000000


DATE=`date '+%d-%m-%Y[%H:%M:%S]'`
mkdir $DATE
cd ./$DATE

n_rep=15

scripts=()

for node_idx in ${!n_nodes[@]}
do
    m_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} $strong_scaling_thin_mn ${weak_scaling_p0_mn[node_idx]} ${weak_scaling_p1_mn[node_idx]})
    n_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} $strong_scaling_thin_mn ${weak_scaling_p0_mn[node_idx]} ${weak_scaling_p1_mn[node_idx]})
    k_values=($strong_scaling_square ${weak_scaling_p0[node_idx]} ${weak_scaling_p1[node_idx]} $strong_scaling_thin_k ${weak_scaling_p0_k[node_idx]} ${weak_scaling_p1_k[node_idx]})

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
        #./$file
        echo "Executing the script "$file
    done
done


