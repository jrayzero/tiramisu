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
    if (rank == 0) {
        Halide::Buffer<uint8_t> image = Halide::Tools::load_image("./images/rgb.png");
        Halide::Buffer<uint8_t> output_buf(image.extent(0) - 8, image.extent(1) - 8, image.channels());
	Halide::Buffer<uint8_t> ref = Halide::Tools::load_image("./images/reference_blurxy.png");

        dtest_03(image.raw_buffer(), output_buf.raw_buffer());
	compare_buffers("dtest_03", output_buf, ref);
	//        Halide::Tools::save_image(output_buf, "./build/dtest_03.png");
	//	std::cerr << "Done" << std::endl;
    } else {
        Halide::Buffer<uint8_t> image(0,0);
        Halide::Buffer<uint8_t> output_buf(0,0);
        dtest_03(image.raw_buffer(), output_buf.raw_buffer());
    }

    return 0;
}
