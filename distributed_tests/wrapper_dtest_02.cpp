//
// Created by Jessica Ray on 10/3/17.
//

#include <mpi.h>
#include <iostream>
#include "Halide.h"
#include "wrapper_dtest_02.h"

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    float _input[1280*768];
    for (int i = 0; i < 1280*768; i++) {
        _input[i] = (float)i;
    }

    Halide::Runtime::Buffer<float> input(_input, {1280,768});
    Halide::Runtime::Buffer<float> output(1280,768);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dtest_02(input, output);

    if (rank == 0) {
        int ctr = 0;
        for (int row = 0; row < 1280; row++) {
            for (int col = 0; col < 768; col++) {
                assert(output(col, row) == (ctr++ + 4));
            }
        }
    }

    MPI_Finalize();
    std::cerr << "Rank " << rank << " is complete!" << std::endl;

    return 0;
}

