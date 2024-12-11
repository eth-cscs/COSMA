#!/bin/bash
# Example mps-wrapper.sh usage:
# > srun --cpu-bind=socket [...] mps-wrapper.sh <cmd>

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
# Launch MPS from a single rank per node
if [ $SLURM_LOCALID -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-cuda-mps-control -d
fi

# set cuda device
numa_nodes=$(hwloc-calc --physical --intersect NUMAnode $(taskset -p $$ | awk '{print "0x"$6}'))
export CUDA_VISIBLE_DEVICES=$numa_nodes
# Run the command
exec numactl --membind=$numa_nodes "$@"
