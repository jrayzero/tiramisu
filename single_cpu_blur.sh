#!/bin/bash

n=$1 # mpi nodes
p=$2 # mpi procs
rank_file=$3
machine_list=$4

rm /tmp/single*

set -e

export MPI_NODES=$n
export MPI_PROCS=$p

make -j 10
make single_cpu_blur_test

echo "mpirun -rankfile $rank_file -np $p -x MPI_PROCS=$p -x MPI_NODES=$n /tmp/single_cpu_blur"
mpirun -rankfile $rank_file -np $p -x MPI_PROCS=$p -x MPI_NODES=$n /tmp/single_cpu_blur

