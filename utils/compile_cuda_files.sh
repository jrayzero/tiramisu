#!/bin/bash

set -e

echo "nvcc -IHalide/include -Iinclude/ -ccbin $NVCC_CLANG --compile -g -O3 --std=c++11 src/tiramisu_cuda.cu -odir build"
nvcc -IHalide/include -Iinclude/ -ccbin $NVCC_CLANG --compile -g -O3 --std=c++11 src/tiramisu_cuda.cu -odir build