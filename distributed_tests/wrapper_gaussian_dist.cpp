//
// Created by Jessica Ray on 11/2/17.
//

#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_gaussian_dist.h"
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

    Halide::Buffer<uint64_t> buff_input = Halide::Buffer<uint64_t>(_COLS, _ROWS / NODES, _CHANNELS);
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    uint64_t *buf = (uint64_t*)malloc(sizeof(uint64_t) * ((_ROWS / NODES)) * _COLS * _CHANNELS);
    unsigned int next = 0;
    for (int y = 0; y < _ROWS / NODES; y++) {
      for (int x = 0; x < _COLS; x++) {
	for (int c = 0; c < _CHANNELS; c++) {
	  buff_input(x, y, c) = next * (rank+1) + c;
	  next++;
	}
      }
    }

    // Generate these on each node as well
    Halide::Buffer<float> kernelX(5);
    Halide::Buffer<float> kernelY(5);

    kernelX(0) = 1.0f; kernelX(1) = 4.0f; kernelX(2) = 6.0f; kernelX(3) = 4.0f; kernelX(4) = 1.0f;
    kernelY(0) = 1.0f/256; kernelY(1) = 4.0f/256; kernelY(2) = 6.0f/256; kernelY(3) = 4.0f/256; kernelY(4) = 1.0f/256;
    Halide::Buffer<uint64_t> buff_output(rank == NODES - 1 ? 0 : _COLS - 4, rank == NODES - 1 ? 0 : _ROWS / NODES, _CHANNELS);
    Halide::Buffer<uint64_t> buff_output_last_node(rank == NODES - 1 ? _COLS - 4 : 0, rank == NODES - 1 ? (_ROWS / NODES) - 4 : 0, _CHANNELS);
    std::cerr << "Rank: " << rank << std::endl;

    Halide::Buffer<uint64_t> buff_gaussian_x(rank == NODES - 1 ? 0 : _COLS - 4, rank == NODES - 1 ? 0 : _ROWS / NODES + 4, rank == NODES - 1 ? 0 : _CHANNELS);
    Halide::Buffer<uint64_t> buff_gaussian_x_last_node(rank == NODES - 1 ? _COLS - 4 : 0, rank == NODES - 1 ? _ROWS / NODES : 0, rank == NODES - 1 ? _CHANNELS : 0);
    
    // Run once to get rid of overhead/any extra compilation stuff that needs to happen
    gaussian_dist(buff_input.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), buff_gaussian_x.raw_buffer(), buff_gaussian_x_last_node.raw_buffer(), buff_output.raw_buffer(), buff_output_last_node.raw_buffer());

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<20; i++) {
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        gaussian_dist(buff_input.raw_buffer(), kernelX.raw_buffer(), kernelY.raw_buffer(), buff_gaussian_x.raw_buffer(), buff_gaussian_x_last_node.raw_buffer(), buff_output.raw_buffer(), buff_output_last_node.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
	/*	if (i == 0) {
	  std::string output_fn = "/data/scratch/jray/tiramisu/build/gaussian_dist_rank_" + std::to_string(rank) + ".txt";
	  std::ofstream myfile;
	  myfile.open (output_fn);
	  if (rank < NODES - 1) {
	    for (int i = 0; i < _ROWS / NODES; i++) {
	      for (int j = 0; j < _COLS - 4; j++) {
		for (int c = 0; c < _CHANNELS; c++) {
		  myfile << buff_output(j, i, c) << std::endl;
		}
	      }
	    }
	  } else {
	    for (int i = 0; i < _ROWS / NODES - 4; i++) {
	      for (int j = 0; j < _COLS - 4; j++) {
		for (int c = 0; c < _CHANNELS; c++) {
		  myfile << buff_output_last_node(j, i, c) << std::endl;
		}
	      }
	    }
	  }
	  myfile.close();
	  }*/
	MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "gaussian_dist", {"Tiramisu_dist"}, {median(duration_vector)});
        free(buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
