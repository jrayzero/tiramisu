//
// Created by Jessica Ray on 11/28/17.
//


#include <iostream>
#include <fstream>
#include <mpi.h>
#include "wrapper_blur.h"
#include "blur_params.h"

int main() {

#ifdef DISTRIBUTE
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    std::vector<std::chrono::duration<double,std::milli>> duration_vector;

#ifdef DISTRIBUTE
    C_LOOP_ITER_TYPE rows_per_node = (C_LOOP_ITER_TYPE)ceil(ROWS/NODES);
#else
    C_LOOP_ITER_TYPE rows_per_node = (C_LOOP_ITER_TYPE)ceil(ROWS);
#endif
    Halide::Buffer<C_DATA_TYPE> buff_input = Halide::Buffer<C_DATA_TYPE>(COLS, rows_per_node);
    C_DATA_TYPE next = 0;
    for (int y = 0; y < rows_per_node; y++) {
        for (int x = 0; x < COLS; x++) {
            buff_input(x,y) = rank * rows_per_node * COLS + next++;
        }
    }

    Halide::Buffer<C_DATA_TYPE> buff_output = Halide::Buffer<C_DATA_TYPE>(COLS - 2, rows_per_node - 2);

    blur_dist(buff_input.raw_buffer(), buff_output.raw_buffer());

#ifdef DISTRIBUTE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    for (int i = 0; i < ITERS; i++) {
        if (rank == 0) {
            std::cerr << "Starting iter: " << i << std::endl;
        }
#ifdef DISTRIBUTE
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        auto start = std::chrono::high_resolution_clock::now();
        blur_dist(buff_input.raw_buffer(), buff_output.raw_buffer());
#ifdef DISTRIBUTE
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
#if defined(PRINT_ITER_0) && !defined(DISTRIBUTE)
        if (i == 0) {
            std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
            std::ofstream myfile;
            myfile.open (output_fn);
            for (int y = 0; y < rows_per_node - 2; y++) {
                for (int x = 0; x < COLS - 2; x++) {
                    myfile << buff_output(x, y) << std::endl;
                }
            }
            myfile.close();
        }
#elif defined(PRINT_ITER_0)
        if (i == 0) {
            std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
            std::ofstream myfile;
            myfile.open (output_fn);
            for (int y = 0; y < rank == NODES - 1 ? rows_per_node - 2 : rows_per_node; y++) {
                for (int x = 0; x < COLS - 2; x++) {
                    myfile << buff_output(x, y) << std::endl;
                }
            }
            myfile.close();
        }
#endif
#ifdef DISTRIBUTE
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "blur_dist", {"Tiramisu_dist"}, {median(duration_vector)});
    }

#ifdef DISTRIBUTE
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    return 0;

}
