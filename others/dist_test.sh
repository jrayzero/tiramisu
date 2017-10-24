#!/bin/bash

set -e

num_nodes=$1
test=$2

echo "srun -t 10 --exclusive --cpu_bind=verbose,cores make dtests"
srun -t 10 --exclusive --cpu_bind=verbose,cores make dtests

#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#do
echo "time srun -N $num_nodes -t 10 --exclusive --cpu_bind=verbose,cores ./build/dtest_$test"
time srun -N $num_nodes -t 10 --exclusive --cpu_bind=verbose,cores ./build/dtest_$test
#done
