#!/bin/bash

set -e

num_nodes=$1

echo "srun -t 10 --exclusive --cpu_bind=verbose,cores make dtests"
srun -t 10 --exclusive --cpu_bind=verbose,cores make dtests

for i in 01 02
do
    echo "srun -N $num_nodes -t 10 --exclusive --cpu_bind=verbose,cores ./build/dtest_$i"
    srun -N $num_nodes -t 10 --exclusive --cpu_bind=verbose,cores ./build/dtest_$i
done
