//
// Created by Jessica Ray on 11/2/17.
//

#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_dblurxy_dist_data.h"
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
    uint64_t *buf = (uint64_t*)malloc(sizeof(uint64_t) * ((_ROWS / _NODES) + 2) * _COLS);
    unsigned int next = 0;
    for (int y = 0; y < _ROWS / _NODES; y++) {
      for (int x = 0; x < _COLS; x++) {
	buf[next] = next;
	next++;
      }
    }

    Halide::Buffer<uint64_t> buff_input = Halide::Buffer<uint64_t>(buf, {_COLS, _ROWS / _NODES + 2});
    Halide::Buffer<uint64_t> buff_bx(_COLS - 2, (_ROWS / _NODES) + 2);
    Halide::Buffer<uint64_t> buff_bx_last_node(rank == _NODES - 1 ? _COLS - 2 : 0,
                                                   rank == _NODES - 1 ? (_ROWS / _NODES) : 0);
    Halide::Buffer<uint64_t> buff_by(_COLS - 2, _ROWS / _NODES);
    Halide::Buffer<uint64_t> buff_by_last_node(rank == _NODES - 1 ? _COLS - 2 : 0,
                                                   rank == _NODES - 1 ? (_ROWS / _NODES) - 2 : 0);
    std::cerr << "Rank: " << rank << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<10; i++) {
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
        auto start = std::chrono::high_resolution_clock::now();
        dblurxy_dist_data(buff_input.raw_buffer(), buff_bx.raw_buffer(), buff_bx_last_node.raw_buffer(), buff_by.raw_buffer(), buff_by_last_node.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
    }

    if (rank == 0) {
        free(buf);
        print_time("performance_CPU.csv", "blurxy_dist", {"Tiramisu_dist"}, {median(duration_vector)});
    }

    // print out the results to file
    std::string output_fn = "/data/scratch/jray/tiramisu/build/dblurxy_dist_data_rank_" + std::to_string(rank) + ".txt";
    std::ofstream myfile;
    myfile.open (output_fn);
    for (int i = 0; i < (_ROWS / _NODES) - (rank == _NODES - 1 ? 2 : 0); i++) {
      for (int j = 0; j < _COLS - 2; j++) {
	if (rank == _NODES - 1) {
	  myfile << buff_by_last_node(j, i) << std::endl;
	} else {
	  myfile << buff_by(j, i) << std::endl;
	}
      }
    }
    myfile.close();

    MPI_Finalize();

    return 0;
}
