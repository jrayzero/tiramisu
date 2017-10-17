//
// Created by Jessica Ray on 10/3/17.
//

#include <mpi.h>
#include <iostream>
#include "Halide.h"
#include "wrapper_dtest_02.h"

#define ROWS 1280
#define COLS 768

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    float _input[ROWS*COLS];
    for (int i = 0; i < ROWS*COLS; i++) {
        _input[i] = (float)i;
    }

    Halide::Runtime::Buffer<float> input(_input, {COLS, ROWS});
    Halide::Runtime::Buffer<float> output(COLS, ROWS);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dtest_02(input, output);

    if (rank == 0) {
        int ctr = 0;
        for (int row = 0; row < ROWS; row++) {
            for (int col = 0; col < COLS; col++) {
                assert(output(col, row) == (ctr++ + 10));
            }
        }
    }

    MPI_Finalize();
    std::cerr << "Rank " << rank << " is complete!" << std::endl;

    return 0;
}

