//
// Created by Jessica Ray on 11/2/17.
//

#include "wrapper_dblurxy.h"
#include "Halide.h"
#include "halide_image_io.h"
#include "tiramisu/utils.h"
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include "sizes.h"

#define REQ MPI_THREAD_FUNNELED

int main() {

    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    uint64_t *buf = nullptr;
    int height = _ROWS;
    int width = _COLS;
    if (rank == 0) {
        unsigned int next = 0;
        std::cerr << "Filling matrix" << std::endl;
        buf = (uint64_t*)malloc(sizeof(uint64_t) * height * width);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                buf[next++] = random();
            }
        }
        std::cerr << "Done filling matrix" << std::endl;
    } else {
        buf = (uint64_t*)malloc(sizeof(uint64_t));
    }

    Halide::Buffer<uint64_t> image = rank == 0 ?
                                     Halide::Buffer<uint64_t>(buf, {_COLS, _ROWS}) : Halide::Buffer<uint64_t>(0, 0);
    Halide::Buffer<uint64_t> buff_bx((int)_COLS-6, _ROWS);
    Halide::Buffer<uint64_t> buff_input_temp(rank != 0 ? (int)_COLS : 0, rank != 0 ? _ROWS/20 : 0);
    Halide::Buffer<uint64_t> buff_bx_inter((int)_COLS-6, _ROWS/20);
    Halide::Buffer<uint64_t> buff_by_inter((int)_COLS-8, _ROWS/20);
    Halide::Buffer<uint64_t> output(rank == 0 ? image.extent(0) - 8 : 0, rank == 0 ? image.extent(1) - 8 : 0);

    for (int i=0; i<10; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        dblurxy(image.raw_buffer(), buff_bx.raw_buffer(), buff_input_temp.raw_buffer(), buff_bx_inter.raw_buffer(),
                buff_by_inter.raw_buffer(), output.raw_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end - start;
        if (rank == 0) {
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
    }

    if (rank == 0) {
        free(buf);
        print_time("performance_CPU.csv", "blurxy_dist", {"Tiramisu_dist"}, {median(duration_vector)});
    }


    MPI_Finalize();

    return 0;
}