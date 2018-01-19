#!/bin/bash

n=$1 # mpi nodes
p=$2 # mpi procs
rank_file=$3

rm /tmp/*gemv*

set -e

export MPI_NODES=$n
export MPI_PROCS=$p

make -j 10
make gemv

echo "mpirun -rankfile $rank_file -np $p -x MPI_PROCS=$p -x MPI_NODES=$n /tmp/gemv"
mpirun -rankfile $rank_file -np $p -x MPI_PROCS=$p -x MPI_NODES=$n /tmp/gemv
