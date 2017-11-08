#!/bin/bash


rm build/*dist_data*

n=$1

set -e

echo "srun -t 10 --exclusive make dblurxy_dist_data"
srun -t 10 --exclusive make dblurxy_dist_data

echo "srun --nodes=$n -t 100 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/dblurxy_dist_data"
srun --nodes=$n -t 100 --exclusive --ntasks=$n --cpus-per-task=40 --ntasks-per-node=1 ./build/dblurxy_dist_data
