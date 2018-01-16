//
// Created by Jessica Ray on 11/28/17.
//


#include <iostream>
#include <fstream>
#include <mpi.h>
#include "tiramisu/tiramisu_cuda.h"
#include "tiramisu/tiramisu_cuda_runtime.h"
#include "wrapper_blur.h"
#include "blur_params.h"
#include "cuda.h"
#include "Halide.h"
#include <math.h>
#ifdef GPU_ONLY
#include "/tmp/tiramisu_CUDA_kernel_bx.cu.h"
#include "/tmp/tiramisu_CUDA_kernel_by.cu.h"
#endif

extern void clear_static_var_tiramisu_CUDA_kernel_bx();
extern void clear_static_var_tiramisu_CUDA_kernel_by();

int mpi_init() {
  int provided = -1;
  MPI_Init_thread(NULL, NULL, REQ, &provided);
  assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

void init_gpu(int rank) {
  //  if (rank < NODES) {
  //    tiramisu_init_cuda(1);
  //  } else {
  //    tiramisu_init_cuda(1);
  //}
  if (PROCS == 2 && rank == 1) {
    tiramisu_init_cuda(2);
  } else {
    tiramisu_init_cuda(rank % 4);
  }
}

void reset_gpu() {
  cuCtxDestroy(cvars.ctx);
}

void check_results() {
    // combine the rank files together
    C_DATA_TYPE *got = (C_DATA_TYPE*)malloc(sizeof(C_DATA_TYPE) * (ROWS) * (COLS));
    int idx = 0;
    for (int n = 0; n < PROCS; n++) {
      std::ifstream in_file;
      in_file.open("./build/blur_dist_rank_" + std::to_string(n) + ".txt");
      std::string line;
      while(std::getline(in_file, line)) {
        got[idx++] = (C_DATA_TYPE)std::stof(line);
      }
      in_file.close();
    }
    int next = 0;
    Halide::Buffer<C_DATA_TYPE> full_input = Halide::Buffer<C_DATA_TYPE>(COLS, ROWS);
    for (int y = 0; y < ROWS; y++) {
      for (int x = 0; x < COLS; x++) {
        full_input(x, y) = (next++ % 1000);
      }
    }
    idx = 0;
    std::cerr << "Comparing" << std::endl;
    for (int r = 0; r < ROWS - 2; r++) {
      for (int c = 0; c < COLS - 2; c++) {
        C_DATA_TYPE should_be = std::floor((C_DATA_TYPE)((full_input(c,r) + full_input(c+1, r) + full_input(c+2, r) + full_input(c, r+1) +
                                                          full_input(c, r+2) + full_input(c+1, r+1) + full_input(c+1, r+2) + full_input(c+2, r+1) +
                                                          full_input(c+2, r+2)) / (C_DATA_TYPE)9));
        C_DATA_TYPE is = std::floor(got[r * COLS + c]);
        if (std::fabs(should_be - is) > 0.0f) {
          std::cerr << "Mismatch at row " << r << " column " << c << ". Should be " << should_be << ", but is " << is << std::endl;
          assert(false);
        }
      }
    }
    free(got);
}

int main() {
#ifdef GPU_ONLY
  int rank = mpi_init();
    std::cerr << "Running CPU version" << std::endl;
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  C_LOOP_ITER_TYPE rows_per_proc = (C_LOOP_ITER_TYPE)ceil(ROWS/PROCS); 

  for (int i = 0; i < ITERS; i++) {
    init_gpu(rank);    
    halide_buffer_t buff_input;
    C_DATA_TYPE *pinned_host;
#ifdef CHECK_RESULTS
    size_t bytes = COLS * rows_per_proc * sizeof(C_DATA_TYPE);//((rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2) * sizeof(C_DATA_TYPE);
#else
    // simulate using a circular buffer. we copy one row at a time anyway
    size_t bytes = COLS * sizeof(C_DATA_TYPE);
#endif
    std::cerr << "Pinning input" << std::endl;
    assert(cuMemHostAlloc((void**)&pinned_host, bytes, CU_MEMHOSTALLOC_PORTABLE) == 0);
    buff_input.host = (uint8_t*)pinned_host;
    halide_buffer_t buff_output;
    C_DATA_TYPE *pinned_host_out;
#ifdef CHECK_RESULTS
    bytes = COLS * rows_per_proc * sizeof(C_DATA_TYPE);
#else
    bytes = COLS * sizeof(C_DATA_TYPE);
#endif
    std::cerr << "Pinning output" << std::endl;
    assert(cuMemHostAlloc((void**)&pinned_host_out, bytes, CU_MEMHOSTALLOC_PORTABLE) == 0);
    buff_output.host = (uint8_t*)pinned_host_out;
#ifdef CHECK_RESULTS
    std::cerr << "Filling buff_input"  << std::endl;
    int next = 0;
    for (size_t y = 0; y < (size_t)rows_per_proc-2; y++) {
      for (size_t x = 0; x < (size_t)COLS; x++) {
        pinned_host[y * (size_t)COLS + x] = (C_DATA_TYPE)(next++ % 1000);
      }
    }
    //    if (rank < NODES && NODES != 1) { // need to fill up the last two rows with the first two rows of next rank on the machine. We'll just assume we can algorithmically generate it here
      next = 0;
      for (size_t y = rows_per_proc-2; y < (size_t)(rows_per_proc); y++) {
        for (size_t x = 0; x < COLS; x++) {
          pinned_host[y * (size_t)COLS + x] = (C_DATA_TYPE)(next++ % 1000);
        }
      }      
      //    }
#endif // otherwise, don't really care about the actual values b/c we aren't concerned with the filling time
    MPI_Barrier(MPI_COMM_WORLD);
    assert(cuCtxSynchronize() == 0);
    if (rank == 0) {
      std::cerr << "Starting iter: " << i << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    blur_dist_gpu(&buff_input, &buff_output);
    MPI_Barrier(MPI_COMM_WORLD);
    assert(cuCtxSynchronize() == 0);        
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
      std::chrono::duration<double,std::milli> duration = end - start;
      duration_vector.push_back(duration);
      std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
    }
#if defined(CHECK_RESULTS) && defined(DISTRIBUTE)
    if (i == 0) {
      std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
      std::ofstream myfile;
      myfile.open(output_fn);
      for (size_t y = 0; y < ((rank == PROCS - 1) ? (size_t)rows_per_proc-2 : (size_t)rows_per_proc); y++) {
        for (size_t x = 0; x < (size_t)COLS; x++) {
          myfile << pinned_host_out[y * (size_t)COLS + x] << std::endl;
        }
      }
      myfile.close();
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    std::cerr << "Calling synchronize" << std::endl;
    assert(cuCtxSynchronize() == 0);
    assert(cuMemFreeHost(pinned_host) == 0);
    assert(cuMemFreeHost(pinned_host_out) == 0);
    clear_static_var_tiramisu_CUDA_kernel_bx();
    clear_static_var_tiramisu_CUDA_kernel_by();
    reset_gpu();
    MPI_Barrier(MPI_COMM_WORLD);
  }
    
  if (rank == 0) {
    print_time("performance_CPU.csv", "blur_dist", {"Tiramisu_dist"}, {median(duration_vector)});
    std::cout.flush();
        
#ifdef CHECK_RESULTS
    check_results();
#endif
  }
  std::cerr << "DONE with rank " << rank << std::endl;
  MPI_Finalize();
#else // CPU version
  int rank = mpi_init();
  if (rank == 0) {
    std::cerr << "Running CPU version" << std::endl;
  }
  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  C_LOOP_ITER_TYPE rows_per_proc = (C_LOOP_ITER_TYPE)ceil(ROWS/PROCS); 

  for (int i = 0; i < ITERS; i++) {
    halide_buffer_t buff_input;
    C_DATA_TYPE *pinned_host;
#ifdef CHECK_RESULTS
    size_t bytes = COLS * rows_per_proc * sizeof(C_DATA_TYPE);//((rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2) * sizeof(C_DATA_TYPE);
#else
    // simulate using a circular buffer. we copy one row at a time anyway
    size_t bytes = COLS * sizeof(C_DATA_TYPE);
#endif
    std::cerr << "Pinning input" << std::endl;
    pinned_host = (C_DATA_TYPE*)malloc(bytes);
    buff_input.host = (uint8_t*)pinned_host;
    halide_buffer_t buff_output;
    C_DATA_TYPE *pinned_host_out;
#ifdef CHECK_RESULTS
    bytes = COLS * rows_per_proc * sizeof(C_DATA_TYPE);
#else
    bytes = COLS * sizeof(C_DATA_TYPE);
#endif
    std::cerr << "Pinning output" << std::endl;
    pinned_host_out = (C_DATA_TYPE*)malloc(bytes);
    buff_output.host = (uint8_t*)pinned_host_out;
    Halide::Buffer<C_DATA_TYPE> buff_bx = Halide::Buffer<C_DATA_TYPE>(COLS, rows_per_proc);//(rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2);   
#ifdef CHECK_RESULTS
    std::cerr << "Filling buff_input"  << std::endl;
    int next = 0;
    for (size_t y = 0; y < (size_t)rows_per_proc-2; y++) {
      for (size_t x = 0; x < (size_t)COLS; x++) {
        pinned_host[y * (size_t)COLS + x] = (C_DATA_TYPE)(next++ % 1000);
      }
    }
    //    if (rank < NODES && NODES != 1) { // need to fill up the last two rows with the first two rows of next rank on the machine. We'll just assume we can algorithmically generate it here
      next = 0;
      for (size_t y = rows_per_proc-2; y < (size_t)(rows_per_proc); y++) {
        for (size_t x = 0; x < COLS; x++) {
          pinned_host[y * (size_t)COLS + x] = (C_DATA_TYPE)(next++ % 1000);
        }
      }      
      //    }
#endif // otherwise, don't really care about the actual values b/c we aren't concerned with the filling time
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cerr << "Starting iter: " << i << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    blur_dist(&buff_input, &buff_bx, &buff_output);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
      std::chrono::duration<double,std::milli> duration = end - start;
      duration_vector.push_back(duration);
      std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
    }
#if defined(CHECK_RESULTS) && defined(DISTRIBUTE)
    if (i == 0) {
      std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
      std::ofstream myfile;
      myfile.open(output_fn);
      for (size_t y = 0; y < ((rank == PROCS - 1) ? (size_t)rows_per_proc-2 : (size_t)rows_per_proc); y++) {
        for (size_t x = 0; x < (size_t)COLS; x++) {
          myfile << pinned_host_out[y * (size_t)COLS + x] << std::endl;
        }
      }
      myfile.close();
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    free(pinned_host);
    free(pinned_host_out);
    MPI_Barrier(MPI_COMM_WORLD);
  }
    
  if (rank == 0) {
    print_time("performance_CPU.csv", "blur_dist", {"Tiramisu_dist"}, {median(duration_vector)});
    std::cout.flush();
        
#ifdef CHECK_RESULTS
    check_results();
#endif
  }
  std::cerr << "DONE with rank " << rank << std::endl;
  MPI_Finalize();

#endif
  return 0;

}
