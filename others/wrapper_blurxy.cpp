#include "wrapper_blurxy.h"
#include "../benchmarks.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include "../../t_blur_sizes.h"

int main(int, char**) {
  //  std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;
  
    uint8_t *buf = (uint8_t*)malloc(sizeof(uint8_t) * ROWS * COLS * CHANNELS);
      uint8_t v = 0;
      size_t next = 0;
      for (size_t r = 0; r < ROWS; r++) {
	for (size_t c = 0; c < COLS; c++) {
	  for (size_t chann = 0; chann < CHANNELS; chann++) {
	    //	    image.raw_buffer()->host[/*chann * ROWS * COLS + r * COLS + c*/next++] = v++;//(c,r, chann) = v++; // don't care about overflow. whatever
	    buf[next++] = v++;
	  }
	}
      }  

    Halide::Buffer<uint8_t> image(buf, {COLS, ROWS, CHANNELS});
  
  //  Halide::Buffer<uint8_t> output1(image.width()-8, image.height()-8, image.channels());
  Halide::Buffer<uint8_t> output2(image.width()-8, image.height()-8, image.channels());
  
  // Tiramisu
  /*    for (int i=0; i<NB_TESTS; i++)
	{
        auto start1 = std::chrono::high_resolution_clock::now();
        blurxy_tiramisu(image.raw_buffer(), output1.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
	}*/

  // Reference
  for (int i=0; i<10; i++) {
	std::cerr << "starting iter " << i << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    blurxy_ref(image.raw_buffer(), output2.raw_buffer());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration2 = end2 - start2;
    duration_vector_2.push_back(duration2);
	std::cerr << "iter " << i << " is done with time " << duration2.count() << std::endl;
  }

  print_time("performance_CPU.csv", "blurxy",
      	     {/*"Tiramisu", */"Halide"},
      	     {/*median(duration_vector_1), */median(duration_vector_2)});

//  compare_2_2D_arrays("Blurxy",  output1.data(), output2.data(), image.extent(0), image.extent(1));

//    Halide::Tools::save_image(output1, "./build/blurxy_tiramisu.png");
//    Halide::Tools::save_image(output2, "./build/blurxy_ref.png");
  free(buf);
    return 0;
}
