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

void init_gpu(int rank) {
  if (rank < NODES) {
    tiramisu_init_cuda(1);
  } else {
    tiramisu_init_cuda(1);
  }
}

void reset_gpu() {
  cuCtxDestroy(cvars.ctx);
}

int main() {
  int provided = -1;
  MPI_Init_thread(NULL, NULL, REQ, &provided);
  assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<std::chrono::duration<double,std::milli>> duration_vector;
  C_LOOP_ITER_TYPE rows_per_proc = (C_LOOP_ITER_TYPE)ceil(ROWS/PROCS);
  
  for (int i = 0; i < ITERS; i++) {
    init_gpu(rank);    
    halide_buffer_t buff_input;
    C_DATA_TYPE *pinned_host;
    size_t bytes = COLS * ((rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2) * sizeof(C_DATA_TYPE);
    std::cerr << "bytes: " << bytes << std::endl;
    assert(cuMemHostAlloc((void**)&pinned_host, bytes, CU_MEMHOSTALLOC_PORTABLE) == 0);
    buff_input.host = (uint8_t*)pinned_host;
    halide_buffer_t buff_output;
    C_DATA_TYPE *pinned_host_out;
    bytes = COLS * rows_per_proc * sizeof(C_DATA_TYPE);
    std::cerr << "bytes: " << bytes << std::endl;
    assert(cuMemHostAlloc((void**)&pinned_host_out, bytes, CU_MEMHOSTALLOC_PORTABLE) == 0);
    buff_output.host = (uint8_t*)pinned_host_out;
    Halide::Buffer<C_DATA_TYPE> buff_bx = Halide::Buffer<C_DATA_TYPE>(COLS, (rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2);   
#ifdef CHECK_RESULTS
    std::cerr << "Filling buff_input"  << std::endl;
    int next = 0;
    for (int y = 0; y < rows_per_proc; y++) {
      for (int x = 0; x < COLS; x++) {
        pinned_host[y * COLS + x] = (C_DATA_TYPE)(next++ % 1000);
      }
    }
    if (rank < NODES && NODES != 1) { // need to fill up the last two rows with the first two rows of next rank on the machine. We'll just assume we can algorithmically generate it here
      next = 0;
      for (int y = rows_per_proc; y < rows_per_proc + 2; y++) {
        for (int x = 0; x < COLS; x++) {
          pinned_host[y * COLS + x] = (C_DATA_TYPE)(next++ % 1000);
        }
      }      
    }
#endif // otherwise, don't really care about the actual values b/c we aren't concerned with the filling time
    MPI_Barrier(MPI_COMM_WORLD);
    assert(cuCtxSynchronize() == 0);
    if (rank == 0) {
      std::cerr << "Starting iter: " << i << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    blur_dist_gpu(&buff_input, /*buff_input.raw_buffer(), buff_bx.raw_buffer(), buff_output.raw_buffer(),*/ &buff_output);//, buff_wait.raw_buffer());
    //        buff_output.raw_buffer()->set_device_dirty(false);
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
      for (int y = 0; y < /*((rank == PROCS - 1) ? */(rows_per_proc) /*: rows_per_proc)*/; y++) {
        for (int x = 0; x < COLS; x++) {
          myfile << pinned_host_out[y * (COLS) + x] << std::endl;
        }
      }
      myfile.close();
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    std::cerr << "Calling synchronize" << std::endl;
    assert(cuCtxSynchronize() == 0);
    cuMemFreeHost(pinned_host);
    cuMemFreeHost(pinned_host_out);
    reset_gpu();
  }
    
  if (rank == 0) {
    print_time("performance_CPU.csv", "blur_dist", {"Tiramisu_dist"}, {median(duration_vector)});
    std::cout.flush();
        
#if defined(CHECK_RESULTS) && defined(DISTRIBUTE)
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
        C_DATA_TYPE is = std::floor(got[r * COLS + c/*idx++*/]);
        if (std::fabs(should_be - is) > 0.0f) {
          std::cerr << "Mismatch at row " << r << " column " << c << ". Should be " << should_be << ", but is " << is << std::endl;
          assert(false);
        }
      }
    }
    free(got);
#endif
  }
  std::cerr << "DONE with rank " << rank << std::endl;
  MPI_Finalize();
  return 0;

}
