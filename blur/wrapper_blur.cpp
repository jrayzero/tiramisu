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
  return rank;
}

void check_results(float *guess, int rank) {
  // compute the correct output
  float *input = (float*)malloc(sizeof(float) * (ROWS/PROCS+2) * (COLS + 2));
  float *bx = (float*)malloc(sizeof(float) * (ROWS/PROCS + 2) * COLS);
  float *by = (float*)malloc(sizeof(float) * ROWS/PROCS * COLS);
  std::cerr << "Computing truth value" << std::endl;
  float v = 0.0f;//(0.00001f) * rank * ROWS/PROCS * COLS;
  std::cerr << "Truth value input" << std::endl;
  for (int r = 0; r < ROWS/PROCS+2; r++) {
    for (int c = 0; c < COLS; c++) {
      input[r*(COLS+2)+c] = v;
      v += 0.00001f;
    }
    if (r == ROWS/PROCS) {
      v = 0.0f;
    }
  }

  for (int r = 0; r < ROWS/PROCS+2; r++) {
    for (int c = 0; c < COLS; c++) {
      bx[r*COLS+c] = 
        (input[r*(COLS+2)+c] + input[r*(COLS+2)+c+1] + input[r*(COLS+2)+c+2]) / 3.0f;
    }
  }

  for (int r = 0; r < ROWS/PROCS; r++) {
    for (int c = 0; c < COLS; c++) {
      by[r*COLS+c] = 
        (bx[r*COLS+c] + bx[(r+1)*COLS+c] + bx[(r+2)*COLS+c]) / 3.0f;
    }
  }

  // the borders are junk, so ignore those
  std::cerr << "Comparing" << std::endl;
  if (rank != PROCS - 1) {
    for (int r = 0; r < ROWS/PROCS; r++) {
      for (int c = 0; c < COLS - 2; c++) {
        float diff = std::fabs(by[r*COLS+c] - guess[r*COLS+c]);
        if (diff > 0.001f) {
          std::cerr << diff << std::endl;
          std::cerr << "Rank " << rank << " has difference at row " << r << " and col " << c << ". Should be " << by[r*COLS+c] << " but is " << guess[r*COLS+c] << std::endl;
          exit(29);
        }
      }
    }
  } else {
    for (int r = 0; r < ROWS/PROCS - 2; r++) {
      for (int c = 0; c < COLS - 2; c++) {
        float diff = std::fabs(by[r*COLS+c] - guess[r*COLS+c]);
        if (diff > 0.001f) {
          std::cerr << diff << std::endl;
          std::cerr << "Difference at row " << r << " and col " << c << ". Should be " << by[r*COLS+c] << " but is " << guess[r*COLS+c] << std::endl;
          exit(29);
        }
      }
    }
  }
  free(input);
  free(bx);
  free(by);
}

float *generate_blur_input(int rank, bool host = true) {
  assert(ROWS % PROCS == 0);
  int num_rows = ROWS / PROCS;
  float *input;
  // the last rank doesn't need +2 rows, but it's easier to make it uniform
  if (host) {
    CUresult cu = cuMemHostAlloc((void**)&input, sizeof(float)*(COLS+2)*(num_rows+2), CU_MEMHOSTALLOC_PORTABLE);
    size_t total = sizeof(float)*(COLS+2)*(num_rows+2);
    if (cu != CUDA_SUCCESS) {
      std::cerr << "cuMemHostAlloc failed on output with " << cu << std::endl;
      exit(29);
    }
  } else {
    input = (float*)malloc(sizeof(float)*(COLS+2)*(num_rows+2));
  }

#ifdef CHECK
  //  float starting_val = (0.00001f) * rank * num_rows * COLS;
  float starting_val = 0.0f;
  for (int r = 0; r < num_rows+2; r++) { // mimic filling in the data for the next rank
    for (int c = 0; c < COLS; c++) {
      input[r*(COLS+2)+c] = starting_val;
      starting_val += 0.00001f;
    }
    if (r == num_rows) {
      starting_val = 0.0f;
    }
  }
#endif
  return input;
}

void generate_single_cpu_test(int rank) {
#if defined(CPU) && !defined(DIST)
  std::cerr << "Beginning single cpu" << std::endl;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  halide_buffer_t hb_input;
  halide_buffer_t hb_output;
  halide_buffer_t hb_bx;
  std::cerr << "Generating blur input" << std::endl;
  float *input = generate_blur_input(rank, false);
  float *bx = (float*)malloc(sizeof(float)*COLS*(ROWS+2));
  float *output = (float*)malloc(sizeof(float)*COLS*ROWS);
  hb_input.host = (uint8_t*)input;
  hb_output.host = (uint8_t*)output;
  hb_bx.host = (uint8_t*)bx;
  for (int i = 0; i < ITERS; i++) {
    std::cerr << "starting iter " << i << std::endl;
    auto start = std::chrono::high_resolution_clock::now();    
    blur_single_cpu(&hb_input, &hb_bx, &hb_output);
#ifdef CHECK
    std::cerr << "Comparing results" << std::endl;
    check_results(output, rank);
    std::cerr << "Success!" << std::endl;
#endif
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    duration_vector.push_back(duration);
    std::cerr << "Iteration " << i << " complete: " << duration.count() << "ms." << std::endl;
  }
  if (rank == 0) {
    print_time("performance_CPU.csv", "BLUR CPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
  }
    MPI_Finalize();
#endif
}

void generate_multi_cpu_test(int rank) {
#if defined(CPU) && defined(DIST)
  std::cerr << "Beginning multi cpu" << std::endl;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  halide_buffer_t hb_input;
  halide_buffer_t hb_output;
  halide_buffer_t hb_bx;
  std::cerr << "Generating blur input" << std::endl;
  float *input = generate_blur_input(rank, false);
  float *bx = (float*)malloc(sizeof(float)*COLS*(ROWS+2));
  float *output = (float*)malloc(sizeof(float)*COLS*ROWS);
  hb_input.host = (uint8_t*)input;
  hb_output.host = (uint8_t*)output;
  hb_bx.host = (uint8_t*)bx;
  for (int i = 0; i < ITERS; i++) {
    std::cerr << "starting iter " << i << std::endl;
    auto start = std::chrono::high_resolution_clock::now();    
    blur_multi_cpu(&hb_input, &hb_bx, &hb_output);
#ifdef CHECK
    std::cerr << "Comparing results" << std::endl;
    check_results(output, rank);
    std::cerr << "Success!" << std::endl;
#endif
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    duration_vector.push_back(duration);
    std::cerr << "Iteration " << i << " complete: " << duration.count() << "ms." << std::endl;
  }
  if (rank == 0) {
    print_time("performance_CPU.csv", "BLUR CPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
  }
    MPI_Finalize();
#endif
}

void generate_single_gpu_test(int rank) {
#if defined(GPU) && !defined(DIST)
  std::cerr << "Beginning single gpu" << std::endl;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  halide_buffer_t hb_input;
  halide_buffer_t hb_output;
  for (int i = 0; i < ITERS; i++) {
    //    MPI_Barrier(MPI_COMM_WORLD);
    tiramisu_init_cuda(1);
    std::cerr << "Generating blur input" << std::endl;
    float *input = generate_blur_input(rank);
    float *output;
    CUresult cu = cuMemHostAlloc((void**)&output, sizeof(float)*COLS*ROWS, CU_MEMHOSTALLOC_PORTABLE);
    if (cu != CUDA_SUCCESS) {
      std::cerr << "cuMemHostAlloc failed on output with " << cu << std::endl;
      exit(29);
    }
    hb_input.host = (uint8_t*)input;
    hb_output.host = (uint8_t*)output;
    std::cerr << "starting iter " << i << std::endl;
    auto start = std::chrono::high_resolution_clock::now();    
    blur_single_gpu(&hb_input, &hb_output);
    clear_static_var_tiramisu_CUDA_kernel_bx();
    clear_static_var_tiramisu_CUDA_kernel_by();
#ifdef CHECK
    std::cerr << "Comparing results" << std::endl;
    check_results(output, rank);
    std::cerr << "Success!" << std::endl;
#endif
    //    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    duration_vector.push_back(duration);
    std::cerr << "Iteration " << i << " complete: " << duration.count() << "ms." << std::endl;
    cuCtxSynchronize();
    cuCtxDestroy(cvars.ctx);
  }
  if (rank == 0) {
    print_time("performance_CPU.csv", "BLUR GPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
  }
    MPI_Finalize();
#endif
}

void generate_multi_gpu_test(int rank) {
#if defined(GPU) && defined(DIST)
  std::cerr << "Beginning Multi gpu" << std::endl;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  halide_buffer_t hb_input;
  halide_buffer_t hb_output;
  halide_buffer_t hb_d2h_wait;
  for (int i = 0; i < ITERS; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cerr << "Allocating rank 0 with 1" << std::endl;
      tiramisu_init_cuda(1);
    } else {
      std::cerr << "Allocating rank " << rank << " with " << (rank%4) << std::endl;      
      tiramisu_init_cuda(rank % 4);
    }
    std::cerr << "Generating blur input" << std::endl;
    float *input = generate_blur_input(rank);
    float *output;
    CUevent *d2h_wait = (CUevent*)malloc(sizeof(CUevent) * ROWS/PROCS);
    hb_d2h_wait.host = (uint8_t*)d2h_wait;
    CUstream s;
    cuStreamCreate(&s, CU_STREAM_DEFAULT);
    for (int r = 0; r < RESIDENT; r++) {
      cuEventCreate(&d2h_wait[r], 0);
      cuEventRecord(d2h_wait[r], s);
    }
    CUresult cu = cuMemHostAlloc((void**)&output, sizeof(float)*COLS*ROWS/PROCS, CU_MEMHOSTALLOC_PORTABLE);
    if (cu != CUDA_SUCCESS) {
      std::cerr << "cuMemHostAlloc failed on output with " << cu << std::endl;
      exit(29);
    }
    hb_input.host = (uint8_t*)input;
    hb_output.host = (uint8_t*)output;
    if (rank == 0) {
      std::cerr << "starting iter " << i << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();    
    blur_multi_gpu(&hb_input, &hb_output, &hb_d2h_wait);
    clear_static_var_tiramisu_CUDA_kernel_bx();
    clear_static_var_tiramisu_CUDA_kernel_by();
#ifdef CHECK
    std::cerr << "Comparing results" << std::endl;
    check_results(output, rank);
    std::cerr << "Success!" << std::endl;
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    duration_vector.push_back(duration);
    if (rank == 0) {
      std::cerr << "Iteration " << i << " complete: " << duration.count() << "ms." << std::endl;
    }
    cuCtxSynchronize();
    cuCtxDestroy(cvars.ctx);
  }
  if (rank == 0) {
    print_time("performance_CPU.csv", "BLUR GPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
  }
    MPI_Finalize();
#endif
}


int main() {
  int rank = mpi_init();
#ifdef GPU
#ifdef DIST
  generate_multi_gpu_test(rank);
#else
  generate_single_gpu_test(rank);
#endif
#elif defined(CPU)
#ifdef DIST
  generate_multi_cpu_test(rank);
#else
  generate_single_cpu_test(rank);
#endif
#endif
}

