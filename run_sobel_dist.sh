#!/bin/bash

n=$1

rm build/*sobel_dist*

set -e

export MPI_NODES=$n

echo "srun -t 10 --exclusive make sobel_dist"
srun -t 10 --exclusive make sobel_dist

echo "srun --nodes=$n -t 10 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/sobel_dist"
srun --nodes=$n -t 10 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/sobel_dist
