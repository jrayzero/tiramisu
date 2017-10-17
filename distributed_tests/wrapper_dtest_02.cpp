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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      uint32_t _input[ROWS*COLS];
      for (int i = 0; i < ROWS*COLS; i++) {
        _input[i] = (uint32_t)i;
      }
      
      Halide::Runtime::Buffer<uint32_t> input(_input, {COLS, ROWS});
      Halide::Runtime::Buffer<uint32_t> output(COLS, ROWS);
      
      dtest_02(input, output);
      
      int ctr = 0;
      for (int row = 0; row < ROWS; row++) {
	for (int col = 0; col < COLS; col++) {
	  assert(output(col, row) == (ctr++ + 10));
	}
      }
    } else {
      
      Halide::Runtime::Buffer<uint32_t> input(0,0);
      Halide::Runtime::Buffer<uint32_t> output(0,0);
      dtest_02(input, output);
    }

    MPI_Finalize();
    std::cerr << "Rank " << rank << " is complete!" << std::endl;

    return 0;
}

