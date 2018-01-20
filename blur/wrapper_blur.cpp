#include <tiramisu/core.h>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <assert.h>
#include "tiramisu/tiramisu_cuda.h"
#include "Halide.h"
#include "blur.h"
#include "wrapper_blur.h"
#ifdef GPU
#include "/tmp/tiramisu_cuda_runtime.h"
#include "/tmp/tiramisu_CUDA_kernel_bx.cu.h"
#include "/tmp/tiramisu_CUDA_kernel_by.cu.h"
#endif

extern void clear_static_var_tiramisu_CUDA_kernel_bx();
extern void clear_static_var_tiramisu_CUDA_kernel_by();

int mpi_init() {
  int provided = -1;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  assert(provided == MPI_THREAD_FUNNELED && "Did not get the appropriate MPI thread requirement.");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(rank == 0 && "Only one rank allowed right now. Need to adding error check for multirank");
  return rank;
}

void check_results(float *guess) {
  // compute the correct output
  float *input = (float*)malloc(sizeof(float) * ROWS * (COLS + 2));
  float *bx = (float*)malloc(sizeof(float) * (ROWS + 2) * COLS);
  float *by = (float*)malloc(sizeof(float) * ROWS * COLS);
  std::cerr << "Computing truth value" << std::endl;
  float v = 0.0f;
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      input[r*COLS+c] = v;
      v += 0.01f;
    }
  }

  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      bx[r*COLS+c] = 
        (input[r*COLS+c] + input[r*COLS+c+1] + input[r*COLS+c+2]) / 3.0f;
    }
  }

  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      by[r*COLS+c] = 
        (by[r*COLS+c] + by[(r+1)*COLS+c] + by[(r+2)*COLS+c]) / 3.0f;
    }
  }

  // the borders are junk, so ignore those
  std::cerr << "Comparing" << std::endl;
  for (int r = 0; r < ROWS - 2; r++) {
    for (int c = 0; c < COLS - 2; c++) {
      float diff = std::fabs(by[r*COLS+c] - guess[r*COLS+c]);
      if (diff > 0.00001f) {
        std::cerr << "Difference at row " << r << " and col " << c << ". Should be " << by[r*COLS+c] << " but is " << guess[r*COLS+c] << std::endl;
          exit(29);
      }
    }
  }
  free(input);
  free(bx);
  free(by);
}

void generate_blur_input(int rank, float **input) {
  assert(ROWS % MPI_RANKS == 0);
  int num_rows = ROWS / MPI_RANKS;
  // the last rank doesn't need +2 rows, but it's easier to make it uniform
  cuMemHostAlloc((void**)input, sizeof(float)*(COLS+2)*(num_rows+2), CU_MEMHOSTALLOC_PORTABLE);
  float starting_val = (0.01f) * rank * num_rows * COLS;
  for (int r = 0; r < num_rows+2; r++) { // mimic filling in the data for the next rank
    for (int c = 0; c < COLS; c++) {
      input[r*(COLS+2)+c] = starting_val;
      starting_val += 0.01f;
    }
  }
}

void generate_single_gpu_test() {
  int rank = mpi_init();
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  float *input;
  float *output;
  halide_buffer_t hb_input;
  halide_buffer_t hb_output;
  for (int i = 0; i < ITERS; i++) {
    MPI_Barrier();
    tiramisu_init_cuda(rank % 4);
    generate_blur_input(rank, &input);
    output = cuMemHostAlloc((void**)&output, sizeof(float)*COLS*ROWS, CU_MEMHOSTALLOC_PORTABLE);
    hb_input.host = (uint8_t*)input;
    hb_output.host = (uint8_t*)output;
    auto start = std::chrono::high_resolution_clock::now();
    blur_single_gpu(&hb_input, &hb_output);
    clear_static_var_tiramisu_CUDA_kernel_bx();
    clear_static_var_tiramisu_CUDA_kernel_by();
#ifdef CHECK
    check_results(output);
#endif
    cuCtxSynchronize();
    cuCtxDestroy(cvars.ctx);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    duration_vector.push_back(duration);
    std::cerr << "Iteration " << iter << " complete: " << duration.count() << "ms." << std::endl;
  }
  if (rank == 0) {
    print_time("performance_CPU.csv", "GEMV GPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
  }
  MPI_Finalize();
}

int main() {

}

