#include "wrapper_gaussian.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>

int main(int, char**)
{
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("./images/rgb.png");

    Halide::Buffer<float> kernelX(5);
    Halide::Buffer<float> kernelY(5);

    kernelX(0) = 1.0f; kernelX(1) = 4.0f; kernelX(2) = 6.0f; kernelX(3) = 4.0f; kernelX(4) = 1.0f;
    kernelY(0) = 1.0f/256; kernelY(1) = 4.0f/256; kernelY(2) = 6.0f/256; kernelY(3) = 4.0f/256; kernelY(4) = 1.0f/256;

    Halide::Buffer<uint8_t> output1(input.width()-8, input.height()-8, input.channels());
    Halide::Buffer<uint8_t> output2(input.width()-8, input.height()-8, input.channels());

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        gaussian_tiramisu(input.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // Reference
    for (int i=0; i<NB_TESTS; i++)
    {
        auto start2 = std::chrono::high_resolution_clock::now();
        gaussian_ref(input.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), output2.raw_buffer());
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration2 = end2 - start2;
        duration_vector_2.push_back(duration2);
    }

    print_time("performance_CPU.csv", "gaussian",
               {"  Tiramisu "," Halide "},
               {median(duration_vector_1), median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), input.extent(0), input.extent(1));

    Halide::Tools::save_image(output1, "./build/gaussian_tiramisu.png");
    Halide::Tools::save_image(output2, "./build/gaussian_ref.png");

    return 0;
}