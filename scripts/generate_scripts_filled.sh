experiment_time="00:45:00"

n_nodes=(4 7 8 13 16 25 27 32 37 61 64 81 93 128 201 216 256 333 473 512 )
p_range=(16 28 32 52 64 100 108 128 148 244 256 324 372 512 804 864 1024 1332 1892 2048 )
p_rows=(4 4 4 4 8 10 9 8 4 4 16 18 12 16 12 24 32 36 43 32 )
p_cols=(4 7 8 13 8 10 12 16 37 61 16 18 31 32 67 36 32 37 44 64 )
strong_scaling_square=16384

weak_scaling_p0=(29912 39384 42036 53161 58694 72308 74901 80875 86248 106327 108345 118285 124019 136169 146260 146427 144335 131522 85034 67685 )
weak_scaling_p1=(18993 22787 23788 27760 29615 33901 34676 36416 37927 43137 43620 45884 47086 49261 49704 49320 47708 42650 28674 23930 )

strong_scaling_thin_mn=17408
strong_scaling_thin_k=3735552

weak_scaling_p0_mn=(3972 4763 4972 5797 6182 7066 7225 7580 7888 8930 9025 9460 9683 10047 9923 9794 9322 7985 4497 3337 )
weak_scaling_p0_k=(324231 468556 511315 700985 801053 1062403 1114544 1237543 1351659 1807675 1856361 2104857 2256031 2603694 2988940 3021773 3043871 2846015 1801947 1372762 )

weak_scaling_p1_mn=(2838 3198 3289 3635 3789 4123 4180 4306 4411 4732 4758 4866 4910 4933 4674 4587 4313 3653 2142 1668 )
weak_scaling_p1_k=(198569 253467 268490 330574 360766 433528 447140 478290 506029 606686 616607 664436 691280 745634 781135 778844 760512 682971 433043 343170 )

mem_limit=2684354560

DATE=`date '+%d-%m-%Y[%H:%M:%S]'`
mkdir $DATE
cd ./$DATE

n_rep=1

files=()

for node_idx in ${!n_nodes[@]}
do
    if [ $node_idx -le 5 ]
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
        sbatch ./$file
        echo "Executing the script "$file
    done
done


