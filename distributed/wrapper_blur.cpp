//
// Created by Jessica Ray on 11/28/17.
//


#include <iostream>
#include <fstream>
#include <mpi.h>
#include "tiramisu/tiramisu_cuda.h"
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
    
#ifdef GPU
    std::cerr << "My rank is " << rank << std::endl;
    assert(rank <= 1);
    cudaSetDevice(rank);
#endif
    
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;

#ifdef DISTRIBUTE
    //    C_LOOP_ITER_TYPE rows_per_node = (C_LOOP_ITER_TYPE)ceil(ROWS/NODES);
    C_LOOP_ITER_TYPE rows_per_proc = (C_LOOP_ITER_TYPE)ceil(ROWS/PROCS);
#else
    //    C_LOOP_ITER_TYPE rows_per_node = (C_LOOP_ITER_TYPE)ceil(ROWS);
#endif
    Halide::Buffer<C_DATA_TYPE> buff_input = Halide::Buffer<C_DATA_TYPE>(COLS, rows_per_proc + 2);
    C_DATA_TYPE next = 0;
    for (int y = 0; y < rows_per_proc; y++) {
        for (int x = 0; x < COLS; x++) {
          buff_input(x,y) = (C_DATA_TYPE)(rank * rows_per_proc * COLS + next++) + 0.1f;
        }
    }

#ifdef DISTRIBUTE
    Halide::Buffer<C_DATA_TYPE> buff_output = Halide::Buffer<C_DATA_TYPE>(COLS - 2, (rank == PROCS - 1) ? rows_per_proc - 2 : rows_per_proc);
#else
    Halide::Buffer<C_DATA_TYPE> buff_output = Halide::Buffer<C_DATA_TYPE>(COLS - 2, rows_per_proc - 2);
#endif
#ifdef CPU_ONLY
    blur_dist(buff_input.raw_buffer(), buff_output.raw_buffer());
#elif defined(GPU_ONLY)
    // make a temporary gpu buffer for input, bx, and by
    C_DATA_TYPE *dinput, *dbx, *dby;
    int dinput_dim_0 = rows_per_proc + 2;
    int dinput_dim_1 = COLS;
    int dbx_dim_0 = rank == PROCS - 1 ? rows_per_proc : rows_per_proc + 2;
    int dbx_dim_1 = COLS - 2;
    int dby_dim_0 = rank == PROCS - 1 ? rows_per_proc - 2 : rows_per_proc;
    int dby_dim_1 = COLS - 2;
    tiramisu_cuda_malloc((void**)&dinput, dinput_dim_0 * dinput_dim_1 * sizeof(C_DATA_TYPE));
    tiramisu_cuda_malloc((void**)&dbx, dbx_dim_0 * dbx_dim_1 * sizeof(C_DATA_TYPE));
    tiramisu_cuda_malloc((void**)&dby, dby_dim_0 * dby_dim_1 * sizeof(C_DATA_TYPE));
    Halide::Buffer<C_DATA_TYPE> buff_dinput = Halide::Buffer<C_DATA_TYPE>(dinput, {dinput_dim_1, dinput_dim_0});
    Halide::Buffer<C_DATA_TYPE> buff_dbx = Halide::Buffer<C_DATA_TYPE>(dbx, {dbx_dim_1, dbx_dim_0});
    Halide::Buffer<C_DATA_TYPE> buff_dby = Halide::Buffer<C_DATA_TYPE>(dby, {dby_dim_1, dby_dim_0});
    blur_dist_gpu(buff_input.raw_buffer(), buff_dinput.raw_buffer(), buff_dbx.raw_buffer(), buff_dby.raw_buffer(), buff_output.raw_buffer());
#endif
    
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
#ifdef CPU_ONLY
        blur_dist(buff_input.raw_buffer(), buff_output.raw_buffer());
#elif defined(GPU_ONLY)        
        blur_dist_gpu(buff_input.raw_buffer(), buff_dinput.raw_buffer(), buff_dbx.raw_buffer(), buff_dby.raw_buffer(), buff_output.raw_buffer());
#endif
#ifdef DISTRIBUTE
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
#if defined(CHECK_RESULTS) && !defined(DISTRIBUTE)
        if (i == 0) {
            std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
            std::ofstream myfile;
            myfile.open(output_fn);
            for (int y = 0; y < rows_per_proc - 2; y++) {
                for (int x = 0; x < COLS - 2; x++) {
                    myfile << buff_output(x, y) << std::endl;
                }
            }
            myfile.close();
        }
#elif defined(CHECK_RESULTS)
        if (i == 0) {
            std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
            std::ofstream myfile;
            myfile.open(output_fn);
            for (int y = 0; y < ((rank == PROCS - 1) ? (rows_per_proc - 2) : rows_per_proc); y++) {
                for (int x = 0; x < COLS - 2; x++) {
                    myfile << buff_dby(x, y) << std::endl;
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
        std::cout.flush();
        tiramisu_cuda_free(dinput);
        tiramisu_cuda_free(dbx);
        tiramisu_cuda_free(dby);
        
#if defined(CHECK_RESULTS) && defined(DISTRIBUTE)
 // combine the rank files together
        C_DATA_TYPE *got = (C_DATA_TYPE*)malloc(sizeof(C_DATA_TYPE) * (ROWS - 2) * (COLS - 2));
        int idx = 0;
        for (int n = 0; n < PROCS; n++) {
            std::ifstream in_file;
            in_file.open("./build/blur_dist_rank_" + std::to_string(n) + ".txt");
            std::string line;
            while(std::getline(in_file, line)) {
                got[idx++] = (C_DATA_TYPE)std::stoi(line);
            }
            in_file.close();
        }
        next = 0;
        Halide::Buffer<C_DATA_TYPE> full_input = Halide::Buffer<C_DATA_TYPE>(COLS, ROWS);
        for (int y = 0; y < ROWS; y++) {
            for (int x = 0; x < COLS; x++) {
                full_input(x, y) = next++;
            }
        }
        idx = 0;
	std::cerr << "Comparing" << std::endl;
        for (int r = 0; r < ROWS - 2; r++) {
            for (int c = 0; c < COLS - 2; c++) {
                assert(((full_input(c,r) + full_input(c+1, r) + full_input(c+2, r) + full_input(c, r+1) +
                        full_input(c, r+2) + full_input(c+1, r+1) + full_input(c+1, r+2) + full_input(c+2, r+1) +
                        full_input(c+2, r+2)) / 9) == got[idx++]);
            }
        }
        free(got);

#elif defined(CHECK_RESULTS) // not distributed
        next = 0;
        Halide::Buffer<C_DATA_TYPE> full_input = Halide::Buffer<C_DATA_TYPE>(COLS, ROWS);
        for (int y = 0; y < ROWS; y++) {
            for (int x = 0; x < COLS; x++) {
                full_input(x, y) = next++;
            }
        }
        for (int r = 0; r < ROWS - 2; r++) {
            for (int c = 0; c < COLS - 2; c++) {
                assert(((full_input(c,r) + full_input(c+1, r) + full_input(c+2, r) + full_input(c, r+1) +
                         full_input(c, r+2) + full_input(c+1, r+1) + full_input(c+1, r+2) + full_input(c+2, r+1) +
                         full_input(c+2, r+2)) / 9) == buff_output(c,r));
            }
        }
#endif
}
    std::cerr << "DONE with rank " << rank << std::endl;
#ifdef DISTRIBUTE
    MPI_Finalize();
#endif
    return 0;

}
