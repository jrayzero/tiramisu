//
// Created by Jessica Ray on 10/3/17.
//

#include <mpi.h>
#include "Halide.h"
#include "wrapper_dtest_01.h"

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    float _input[160*160];
    for (int i = 0; i < 160*160; i++) {
        _input[i] = (float)i;
    }

    Halide::Runtime::Buffer<float> input(_input, {160,160});
    Halide::Runtime::Buffer<float> output(160,160);

    dtest01(input, output);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int ctr = 0;
        for (int row = 0; row < 160; row++) {
            for (int col = 0; col < 160; col++) {
                assert(output(col, row) == ctr++);
            }
        }
    }

    MPI_Finalize();

}