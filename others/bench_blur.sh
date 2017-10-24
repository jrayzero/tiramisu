#!/bin/bash

set -e

echo "srun -t 10 --exclusive --cpu_bind=verbose,cores make benchmarks"
srun -t 10 --exclusive --cpu_bind=verbose,cores make benchmarks

#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
#do
#echo "time srun -N 1 -t 10 --exclusive --cpu_bind=verbose,cores ./build/bench_blurxy"
#time srun -N 1 -t 10 --exclusive --cpu_bind=verbose,cores ./build/bench_blurxy
#done
