
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

int main(int, char **)
{
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << rank << std::endl;
    if (rank == 0) {
        Halide::Buffer<uint8_t> image = Halide::Tools::load_image("./images/rgb.png");
        Halide::Buffer<uint8_t> output_buf(image.extent(0) - 8, image.extent(1) - 8, image.channels());
	Halide::Buffer<uint8_t> ref = Halide::Tools::load_image("./images/reference_blurxy.png");
	std::vector<std::chrono::duration<double, std::milli>> duration_vector_1;
	for (int i = 0; i < 20; i++) {
	  auto start1 = std::chrono::high_resolution_clock::now();
	  dtest_03(image.raw_buffer(), output_buf.raw_buffer());
	  auto end1 = std::chrono::high_resolution_clock::now();
	  std::chrono::duration<double,std::milli> duration1 = end1 - start1;
	  duration_vector_1.push_back(duration1);
	  compare_buffers("dtest_03", output_buf, ref);
	}
	print_time("performance_CPU.csv", "blurxy_dist",
		   {"Tiramisu_dist"},
		   {median(duration_vector_1)});

    } else {
      Halide::Buffer<uint8_t> image(0,0); // dummy buffer
      Halide::Buffer<uint8_t> output_buf(0,0); // dummy buffer
      for (int i = 0; i < 20; i++) {
	dtest_03(image.raw_buffer(), output_buf.raw_buffer());
      }
    }

    return 0;
}
