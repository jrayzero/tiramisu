#!/bin/bash

n=$1 # mpi nodes
p=$2 # mpi procs
rank_file=$3
machine_list=$4

set -e

export MPI_NODES=$n
export MPI_PROCS=$p

for machine in $(cat $machine_list)
do
    echo ssh $machine "cd /data/hltemp/jray/tiramisu/; make -j 10; make single_gpu_blur_test"
    ssh $machine "cd /data/hltemp/jray/tiramisu/; make -j 10; make single_gpu_blur_test"
done

#echo "mpirun -rankfile $rank_file -np $p -x MPI_PROCS=$p -x MPI_NODES=$n /tmp/single_gpu_blur"

