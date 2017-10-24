
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

int main(int, char **)
{
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Halide::Buffer<uint8_t> buff_bx(COLS-6, ROWS);
    Halide::Buffer<uint8_t> buff_input_temp(COLS, ROWS);
    Halide::Buffer<uint8_t> buff_bx_inter(COLS-6, ROWS);
    Halide::Buffer<uint8_t> buff_by_inter(COLS-8, ROWS);

    if (rank == 0) {
      // TODO make sure to compare to reference!
      //Halide::Buffer<uint8_t> image = Halide::Tools::load_image("./images/rgb.png");
      uint8_t *buf = (uint8_t*)malloc(sizeof(uint8_t) * ROWS * COLS * CHANNELS);
      uint8_t v = 0;
      size_t next = 0;
      for (size_t r = 0; r < ROWS; r++) {
	for (size_t c = 0; c < COLS; c++) {
	  for (size_t chann = 0; chann < CHANNELS; chann++) {
	    //	    image.raw_buffer()->host[next++] = v++;//(c,r, chann) = v++; // don't care about overflow. whatever
	    buf[next++] = v++;
	  }
	}
      }
      Halide::Buffer<uint8_t> image(buf, {COLS, ROWS, CHANNELS});
      Halide::Buffer<uint8_t> output_buf(image.extent(0) - 8, image.extent(1) - 8, image.channels());
      //	Halide::Buffer<uint8_t> ref = Halide::Tools::load_image("./images/reference_blurxy.png");
      std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
      for (int i = 0; i < 10; i++) {
	std::cerr << "starting iter " << i << std::endl;
	auto start1 = std::chrono::high_resolution_clock::now();
	dtest_03(image.raw_buffer(), buff_bx.raw_buffer(), buff_input_temp.raw_buffer(), buff_bx_inter.raw_buffer(), buff_by_inter.raw_buffer(), output_buf.raw_buffer());
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	duration_vector_1.push_back(duration1);
	std::cerr << "iter " << i << " is done with time " << duration1.count() << std::endl;
	sleep(5);
      }
      //	  compare_buffers("dtest_03", output_buf, ref);
      print_time("performance_CPU.csv", "blurxy_dist",
      		 {"Tiramisu_dist"},
      		 {median(duration_vector_1)});
free(buf);
    } else {
      Halide::Buffer<uint8_t> image(0,0); // dummy buffer
      Halide::Buffer<uint8_t> output_buf(0,0); // dummy buffer
      //      dtest_03(image.raw_buffer(), output_buf.raw_buffer());
      for (int i = 0; i < 10; i++) {
	dtest_03(image.raw_buffer(), buff_bx.raw_buffer(), buff_input_temp.raw_buffer(), buff_bx_inter.raw_buffer(), buff_by_inter.raw_buffer(), output_buf.raw_buffer());
      }
    }

    return 0;
}
