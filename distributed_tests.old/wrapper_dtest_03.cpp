//
// Created by Jessica Ray on 10/17/17.
//

#include "wrapper_dtest_03.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include "../t_blur_sizes.h"

#define REQ MPI_THREAD_FUNNELED

int main(int, char **)
{
  int provided = -1;
  MPI_Init_thread(NULL, NULL, REQ, &provided);
  assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Halide::Buffer<uint8_t> buff_bx(COLS-6, ROWS);
  Halide::Buffer<uint8_t> buff_input_temp(COLS, ROWS);
  Halide::Buffer<uint8_t> buff_bx_inter(COLS-6, ROWS);
  Halide::Buffer<uint8_t> buff_by_inter(COLS-8, ROWS);
  
  Halide::Buffer<uint8_t> *image = nullptr;
  uint8_t *buf = nullptr;
  if (rank == 0) {
    image = (Halide::Buffer<uint8_t>*)malloc(sizeof(Halide::Buffer<uint8_t>));
#ifdef CHECK_RESULTS
   *image = Halide::Tools::load_image("./images/rgb.png");
   std::cerr << "Loaded" << std::endl;
   std::cerr << image->extent(0) << std::endl;
   std::cerr << image->extent(1) << std::endl;
   std::cerr << image->extent(2) << std::endl;
#else
   std::cerr << "Filling matrix" << std::endl;
    buf = (uint8_t*)malloc(sizeof(uint8_t) * ROWS * COLS * CHANNELS);
    uint8_t v = 0;
    size_t next = 0;
    for (size_t r = 0; r < ROWS; r++) {
      for (size_t c = 0; c < COLS; c++) {
        for (size_t chann = 0; chann < CHANNELS; chann++) {
          buf[next++] = v++;
        }
      }
    }
   std::cerr << "Done filling matrix" << std::endl;
    *image = Halide::Buffer<uint8_t>(buf, {COLS, ROWS, CHANNELS});
#endif
  }
  std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
#ifdef CHECK_RESULTS
  Halide::Buffer<uint8_t> ref = Halide::Tools::load_image("./images/reference_blurxy.png");
  std::cerr << "loaded reference" << std::endl;
#endif
  MPI_Barrier(MPI_COMM_WORLD); // make sure matrix is loaded
  for (int iter = 0; iter < ITERS; iter++) {
    if (rank == 0) {
      Halide::Buffer<uint8_t> output_buf(image->extent(0) - 8, image->extent(1) - 8, image->channels());
      std::cerr << "rank " << rank << " is starting iter " << std::endl;
      auto start1 = std::chrono::high_resolution_clock::now();
      dtest_03(image->raw_buffer(), buff_bx.raw_buffer(), buff_input_temp.raw_buffer(), buff_bx_inter.raw_buffer(), buff_by_inter.raw_buffer(), output_buf.raw_buffer());
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
      std::cerr << "iter " << iter << " is done with time " << duration1.count() << std::endl;
#ifdef CHECK_RESULTS
      compare_buffers("dtest_03", output_buf, ref);
#endif
    } else {
      std::cerr << "rank " << rank << " is starting iter " << std::endl;
      Halide::Buffer<uint8_t> image_dummy(0,0); // dummy buffer
      Halide::Buffer<uint8_t> output_dummy(0,0); // dummy buffer
      dtest_03(image_dummy.raw_buffer(), buff_bx.raw_buffer(), buff_input_temp.raw_buffer(), buff_bx_inter.raw_buffer(), buff_by_inter.raw_buffer(), output_dummy.raw_buffer());
      std::cerr << "rank " << rank << " is done with iter " << std::endl;
    }
    // This syncs up all the MPI ranks after running it so we can safely rerun the MPI computation 
    MPI_Barrier(MPI_COMM_WORLD);
    std::cerr << "Rank " << rank << " passed barrier" << std::endl;
  }
  if (rank == 0) {
    print_time("performance_CPU.csv", "blurxy_dist", {"Tiramisu_dist"}, {median(duration_vector_1)});
    free(image);
    free(buf);
  }
  MPI_Finalize();
  return 0;
}