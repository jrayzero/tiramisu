#!/bin/bash

rm /data/hltemp/jray/tiramisu/build/*cvtcolor_gpu_dist*
rm /data/hltemp/jray/tiramisu/build/*cvtcolor_dist*

set -e

export MPI_NODES=24 # this is really mpi processes
export WITH_CUDA=0
echo "make cvtcolor_dist_kitchen_sink"
make cvtcolor_dist_kitchen_sink
export WITH_CUDA=1
echo "make cvtcolor_gpu_dist_kitchen_sink"
make cvtcolor_gpu_dist_kitchen_sink

echo "mpirun -rankfile rank_file -np 12 /data/hltemp/jray/tiramisu/build/cvtcolor_gpu_dist :  -np 12 /data/hltemp/jray/tiramisu/build/cvtcolor_dist"
mpirun -rankfile rank_file -np 12 /data/hltemp/jray/tiramisu/build/cvtcolor_gpu_dist :  -np 12 /data/hltemp/jray/tiramisu/build/cvtcolor_dist


