//
// Created by Jessica Ray on 11/2/17.
//

#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_cvtcolor_kitchen_sink.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "sizes.h"

#define REQ MPI_THREAD_FUNNELED

int main(int ac, char** av) {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank < NODES/2) { 
      int GPU_num = rank % 2 + (rank == 8 || rank == 9 ? 1 : 0);
      std::cerr << "Program " << av[0] << " has rank: " << rank << " is a GPU process that goes onto GPU " << GPU_num << std::endl;
      halide_set_gpu_device(GPU_num);
    } else {
      std::cerr << "Program " << av[0] << " has rank: " << rank << " is a CPU process" << std::endl;   
    }

    //    if (rank <= 15) { // nike 3, 4, 5, and 6
    //      halide_set_gpu_device(rank % 2);
    //    } else if (rank >= 20) { // nike 9
    //      halide_set_gpu_device(rank % 2);
    //    } else { // nike 8
    //      halide_set_gpu_device((rank % 2) + 1); 
    //    }
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    //    std::cerr << "Check rank " << rank << std::endl;
    //    usleep(10000000);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<1; i++) {
        std::cerr << "rank: " << rank << std::endl;
        Halide::Buffer<float> buff_output = Halide::Buffer<float>(COLS, ROWS);
        Halide::Buffer<float> buff_input = Halide::Buffer<float>(COLS, ROWS, 3);
        unsigned int next = 0;
        for (int y = 0; y < ROWS; y++) {
          for (int x = 0; x < COLS; x++) {
            for (int c = 0; c < 3; c++) {
              buff_input(x, y, c) = next + c;
              next++;
            }
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        if (rank < NODES/2) {
          cvtcolor_gpu_dist(buff_input.raw_buffer(), buff_output.raw_buffer());
          buff_output.copy_to_host();
        } else {
          cvtcolor_dist(buff_input.raw_buffer(), buff_output.raw_buffer());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
        if (i == 0) {
          std::string output_fn = "/data/hltemp/jray/tiramisu/build/cvtcolor_kitchen_sink_rank_" + std::to_string(rank) + ".txt";
          std::ofstream myfile;
          myfile.open (output_fn);
          for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS ; j++) {
              myfile << buff_output(j, i) << std::endl;
            }
          }
          myfile.close();
          if (rank < NODES/2) {
            buff_input.device_free();
            buff_output.device_free();
          }
          }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        print_time("performance_GPU.csv", "cvtcolor_ksink", {"Tiramisu_dist"}, {median(duration_vector)});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
