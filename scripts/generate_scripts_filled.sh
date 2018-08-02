experiment_time="00:30:00"

n_nodes=(4 7 8 13 16 25 27 32 37 61 64 81 93 128 201 216 256 333 473 512 )
p_range=(16 28 32 52 64 100 108 128 148 244 256 324 372 512 804 864 1024 1332 1892 2048 )
p_rows=(4 4 4 4 8 10 9 8 4 4 16 18 12 16 12 24 32 36 43 32 )
p_cols=(4 7 8 13 8 10 12 16 37 61 16 18 31 32 67 36 32 37 44 64 )
strong_scaling_square=16384

weak_scaling_p0=(26754 35393 37837 48233 53509 66887 69511 75674 81372 104481 107019 120397 129007 151348 189658 196608 214039 244116 290940 302697 )
weak_scaling_p1=(21235 25590 26754 31454 33709 39115 40132 42470 44576 52660 53509 57880 60608 67418 78362 80264 84941 92723 104230 107019 )

strong_scaling_thin_mn=17408
strong_scaling_thin_k=3735552

weak_scaling_p0_mn=(22796 27470 28721 33766 36186 41990 43081 45592 47852 56530 57442 62134 65063 72373 84121 86163 91184 99537 111890 114885 )
weak_scaling_p0_k=(6405814 9301957 10168482 14054527 16141288 21734456 22878554 25623256 28226512 39392632 40673929 47589994 52182538 64566936 87230002 91516342 102493024 122131059 154326169 162698551 )

weak_scaling_p1_mn=(19541 22129 22796 25393 26592 29364 29871 31020 32037 35802 36186 38131 39320 42212 46665 47417 49242 52205 56440 57442 )
weak_scaling_p1_k=(4707069 6036436 6405814 7948497 8716839 10628878 10999084 11861527 12652044 15800528 16141288 17923112 19058295 21964882 26843526 27715656 29890170 33595509 39267300 40673929 )

mem_limit=5368709120

DATE=`date '+%d-%m-%Y[%H:%M:%S]'`
mkdir $DATE
cd ./$DATE

n_rep=15

files=()

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


