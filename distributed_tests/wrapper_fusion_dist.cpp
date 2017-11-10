//
// Created by Jessica Ray on 11/2/17.
//

#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_fusion_dist.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "sizes.h"

#define REQ MPI_THREAD_FUNNELED
#define CHANNELS 3

int main() {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Halide::Buffer<uint64_t> buff_input = Halide::Buffer<uint64_t>(_COLS, _ROWS / NODES, 3);
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    uint64_t *buf = (uint64_t*)malloc(sizeof(uint64_t) * ((_ROWS / NODES)) * _COLS * CHANNELS);
    unsigned int next = 0;
    for (int y = 0; y < _ROWS / NODES; y++) {
      for (int x = 0; x < _COLS; x++) {
	for (int c = 0; c < CHANNELS; c++) {
	  buff_input(x, y, c) = next * (rank) + c;
	  next++;
	}
      }
    }


    Halide::Buffer<uint64_t> buff_f = Halide::Buffer<uint64_t>(_COLS, _ROWS / NODES, CHANNELS);
    Halide::Buffer<uint64_t> buff_g = Halide::Buffer<uint64_t>(_COLS, _ROWS / NODES, CHANNELS);
    Halide::Buffer<uint64_t> buff_h = Halide::Buffer<uint64_t>(_COLS, _ROWS / NODES, CHANNELS);
    Halide::Buffer<uint64_t> buff_k = Halide::Buffer<uint64_t>(_COLS, _ROWS / NODES, CHANNELS);

    std::cerr << "Rank: " << rank << std::endl;
    
    // Run once to get rid of overhead/any extra compilation stuff that needs to happen
    fusion_dist(buff_input.raw_buffer(), buff_f.raw_buffer(), buff_g.raw_buffer(), buff_h.raw_buffer(), buff_k.raw_buffer());

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<20; i++) {
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        fusion_dist(buff_input.raw_buffer(), buff_f.raw_buffer(), buff_g.raw_buffer(), buff_h.raw_buffer(), buff_k.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
	/*	if (i == 0) {
	  std::string output_fn = "/data/scratch/jray/tiramisu/build/fusion_dist_rank_" + std::to_string(rank) + ".txt";
	  std::ofstream myfile;
	  myfile.open (output_fn);
	  for (int i = 0; i < _ROWS / NODES; i++) {
	    for (int j = 0; j < _COLS ; j++) {
	      for (int c = 0; c < _CHANNELS; c++) {
		myfile << buff_h(j, i, c) << std::endl;
		myfile << buff_k(j, i, c) << std::endl;
	      }
	    }
	  }
	  myfile.close();
	  }*/
	MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "fusion_dist", {"Tiramisu_dist"}, {median(duration_vector)});
        free(buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
