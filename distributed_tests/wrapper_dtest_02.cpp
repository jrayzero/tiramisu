//
// Created by Jessica Ray on 10/6/17.
//

#include "wrapper_dtest_02.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>


#define NN 10

int main(int, char **)
{
    Halide::Buffer<uint8_t> image = Halide::Tools::load_image("./images/rgb.png");
    Halide::Buffer<uint8_t> output_buf(image.extent(0), image.extent(1));

    blurxy(image.raw_buffer(), output_buf.raw_buffer());

    copy_buffer(output_buf, image);
    Halide::Tools::save_image(image, "./build/dtest_02.png");

    return 0;
}
