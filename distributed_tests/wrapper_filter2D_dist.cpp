//
// Created by Jessica Ray on 11/2/17.
//

#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_filter2D_dist.h"
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
    Halide::Buffer<float> buff_input = Halide::Buffer<float>(_COLS, _ROWS / NODES + 2, _CHANNELS);
    float *buf = (float*)malloc(sizeof(float) * (_ROWS / NODES + 2) * _COLS * _CHANNELS);
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
    //    Halide::Buffer<float> kernel(3,3);
    //    kernel(0,0) = 0.0f; kernel(0,1) = 1.0f/5.0f; kernel(0,2) = 0.0f;
    //    kernel(1,0) = 1.0f/5.0f; kernel(1,1) = 1.0f/5.0f; kernel(1,2) = 1.0f/5.0f;
    //    kernel(2,0) = 0.0f; kernel(2,1) = 1.0f; kernel(2,2) = 0.0f;

    Halide::Buffer<float> buff_output(_COLS - 2, rank == NODES - 1 ? _ROWS / NODES - 2 : _ROWS / NODES, _CHANNELS);
    std::cerr << "Rank: " << rank << std::endl;

    // Run once to get rid of overhead/any extra compilation stuff that needs to happen
    filter2D_dist(buff_input.raw_buffer(), buff_output.raw_buffer(), buff_output.raw_buffer());

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<20; i++) {
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        filter2D_dist(buff_input.raw_buffer(), buff_output.raw_buffer(), buff_output.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
	/*        if (i == 0) {
            std::string output_fn = "/data/scratch/jray/tiramisu/build/filter2D_dist_rank_" + std::to_string(rank) + ".txt";
            std::ofstream myfile;
            myfile.open (output_fn);
            if (rank < NODES - 1) {
                for (int i = 0; i < _ROWS / NODES; i++) {
                    for (int j = 0; j < _COLS - 2; j++) {
                        for (int c = 0; c < _CHANNELS; c++) {
                            myfile << buff_output(j, i, c) << std::endl;
                        }
                    }
                }
            } else {
                for (int i = 0; i < _ROWS / NODES - 2; i++) {
                    for (int j = 0; j < _COLS - 2; j++) {
                        for (int c = 0; c < _CHANNELS; c++) {
                            myfile << buff_output(j, i, c) << std::endl;
                        }
                    }
                }
            }
            myfile.close();
	    }*/
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "filter2D_dist", {"Tiramisu_dist"}, {median(duration_vector)});
        free(buf);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
