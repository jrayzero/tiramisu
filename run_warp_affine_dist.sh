#!/bin/bash

n=$1

rm build/*warp_affine_dist*

set -e

export MPI_NODES=$n

echo "srun -t 10 --exclusive make warp_affine_dist"
srun -t 10 --exclusive make warp_affine_dist

echo "srun --nodes=$n -t 100 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/warp_affine_dist"
srun --nodes=$n -t 100 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/warp_affine_dist
