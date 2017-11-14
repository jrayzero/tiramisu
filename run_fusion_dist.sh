#!/bin/bash

n=$1

rm build/*fusion_dist*

set -e

export MPI_NODES=$n

echo "srun -t 10 --exclusive make fusion_dist"
srun -t 10 --exclusive make fusion_dist

echo "srun --nodes=$n -t 100 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/fusion_dist"
srun --nodes=$n -t 100 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/fusion_dist
