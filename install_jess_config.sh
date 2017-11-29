#!/bin/bash

# ISL
cd 3rdParty
rm -rf isl
git clone https://github.com/jrayzero/isl
cd isl
mkdir build
set -e
./autogen.sh
./configure --prefix=$PWD/build/ --with-int=imath
srun make -j 20
srun make install

# Halide
set +e
rm -rf Halide
git clone https://github.com/jrayzero/Halide
cd Halide
set -e
git checkout tiramisu_mpi
srun make -j 20
