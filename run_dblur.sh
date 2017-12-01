#!/bin/bash

n=$1

rm build/*blur*

set -e

export MPI_NODES=$n

echo "srun -t 10 --exclusive make dblur"
srun -t 10 --exclusive make dblur

echo "export LD_LIBRARY_PATH=3rdParty/isl/build/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=3rdParty/isl/build/lib:$LD_LIBRARY_PATH

echo "srun --nodes=$n -t 10 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/blur"
srun --nodes=$n -t 10 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/blur

