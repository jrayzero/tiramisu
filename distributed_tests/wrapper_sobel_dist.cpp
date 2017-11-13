//
// Created by Jessica Ray on 11/2/17.
//

#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_sobel_dist.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "sizes.h"

#define REQ MPI_THREAD_FUNNELED

int main() {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    Halide::Buffer<float> buff_input = Halide::Buffer<float>(_COLS, _ROWS / NODES + 2);
    float *buf = (float*)malloc(sizeof(float) * ((_ROWS / NODES)) * _COLS);
    unsigned int next = 0;
    for (int y = 0; y < _ROWS / NODES; y++) {
      for (int x = 0; x < _COLS; x++) {
    	buff_input(x, y) = next * (rank+1);
    	next++;
      }
    }

    // Generate these on each node as well
    std::cerr << "Rank: " << rank << std::endl;

    //    Halide::Buffer<float> buff_sobel_x(rank == NODES - 1 ? 0 : _COLS - 2, rank == NODES - 1 ? 0 : _ROWS / NODES);
    //    Halide::Buffer<float> buff_sobel_x_last_node(rank == NODES - 1 ? _COLS - 2 : 0, rank == NODES - 1 ? _ROWS / NODES - 2: 0);
    //    Halide::Buffer<float> buff_sobel_y(rank == NODES - 1 ? 0 : _COLS - 2, rank == NODES - 1 ? 0 : _ROWS / NODES);
    //    Halide::Buffer<float> buff_sobel_y_last_node(rank == NODES - 1 ? _COLS - 2 : 0, rank == NODES - 1 ? _ROWS / NODES - 2 : 0);
    Halide::Buffer<float> buff_sobel(rank == NODES - 1 ? 0 : _COLS - 2, rank == NODES - 1 ? 0 : _ROWS / NODES);
    Halide::Buffer<float> buff_sobel_last_node(rank == NODES - 1 ? _COLS - 2 : 0, rank == NODES - 1 ? _ROWS / NODES - 2 : 0);
    // Run once to get rid of overhead/any extra compilation stuff that needs to happen
    sobel_dist(buff_input.raw_buffer(), /*buff_sobel_x.raw_buffer(), buff_sobel_x_last_node.raw_buffer(), buff_sobel_y.raw_buffer(), buff_sobel_y_last_node.raw_buffer(),*/ buff_sobel.raw_buffer(), buff_sobel_last_node.raw_buffer());

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<20; i++) {
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
	sobel_dist(buff_input.raw_buffer(), /*buff_sobel_x.raw_buffer(), buff_sobel_x_last_node.raw_buffer(), buff_sobel_y.raw_buffer(), buff_sobel_y_last_node.raw_buffer(),*/ buff_sobel.raw_buffer(), buff_sobel_last_node.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
	/*	if (i == 0) {
	  std::string output_fn = "/data/scratch/jray/tiramisu/build/sobel_dist_rank_" + std::to_string(rank) + ".txt";
	  std::ofstream myfile;
	  myfile.open (output_fn);
	  if (rank < NODES - 1) {
	    for (int i = 0; i < _ROWS / NODES; i++) {
	      for (int j = 0; j < _COLS - 2; j++) {
		myfile << buff_sobel(j, i) << std::endl;
	      }
	    }
	  } else {
	    for (int i = 0; i < _ROWS / NODES - 2; i++) {
	      for (int j = 0; j < _COLS - 2; j++) {
		myfile << buff_sobel_last_node(j, i) << std::endl;
	      }
	    }
	  }
	  myfile.close();
	  }*/
	MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "sobel_dist", {"Tiramisu_dist"}, {median(duration_vector)});
	free(buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
