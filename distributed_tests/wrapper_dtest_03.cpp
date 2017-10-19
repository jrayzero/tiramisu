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
    std::cerr << "My rank is " << rank << std::endl;
    if (rank == 0) {
        Halide::Buffer<uint8_t> image = Halide::Tools::load_image("./images/rgb.png");
        Halide::Buffer<uint8_t> output_buf(image.extent(0) - 8, image.extent(1) - 8);

        // The blurxy takes a halide_buffer_t * as argument, when "image"
        // is passed, its buffer is actually extracted and passed
        // to the function (c++ operator overloading).
        dtest_03(image.raw_buffer(), output_buf.raw_buffer());
	std::cerr << "This rank is done " << rank << std::endl;
        // TODO(psuriana): not sure why we have to copy the output to image, then
        // write it to file, as opposed to write the output to file directly.
        copy_buffer(output_buf, image);
        Halide::Tools::save_image(image, "./build/dtest_03.png");
    } else {
        Halide::Buffer<uint8_t> image(0,0);
        Halide::Buffer<uint8_t> output_buf(0,0);
        dtest_03(image.raw_buffer(), output_buf.raw_buffer());
	std::cerr << "This rank is done " << rank << std::endl;
    }

    return 0;
}
