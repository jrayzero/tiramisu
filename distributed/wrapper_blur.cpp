//
// Created by Jessica Ray on 11/28/17.
//


#include <iostream>
#include <fstream>
#include <mpi.h>
#include "tiramisu/tiramisu_cuda.h"
#include "wrapper_blur.h"
#include "blur_params.h"
#include "cuda.h"
#include "Halide.h"
#include <math.h>

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

#ifdef GPU_ONLY
    std::cerr << "My rank is " << rank << std::endl;
    if (rank < NODES) {
      tiramisu_init_cuda(1);
    } else {
      tiramisu_init_cuda(1);
    }
#endif
    
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;

#ifdef DISTRIBUTE
    C_LOOP_ITER_TYPE rows_per_proc = (C_LOOP_ITER_TYPE)ceil(ROWS/PROCS);
#else
    C_LOOP_ITER_TYPE rows_per_proc = (C_LOOP_ITER_TYPE)ROWS;
#endif

    Halide::Buffer<C_DATA_TYPE> buff_input = Halide::Buffer<C_DATA_TYPE>(COLS, (rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2);

#ifdef DISTRIBUTE
    Halide::Buffer<C_DATA_TYPE> buff_output = Halide::Buffer<C_DATA_TYPE>(COLS, (rank == PROCS - 1) ? rows_per_proc : rows_per_proc);
    Halide::Buffer<C_DATA_TYPE> buff_bx = Halide::Buffer<C_DATA_TYPE>(COLS, (rank == PROCS - 1) ? rows_per_proc : rows_per_proc + 2);   
    //    Halide::Buffer<C_DATA_TYPE> buff_wait = Halide::Buffer<C_DATA_TYPE>(COLS, rows_per_proc + 2);
#ifdef GPU_ONLY
    //    std::cerr << "Allocating buff_input on GPU" << std::endl;
    //    buff_input.device_malloc(halide_cuda_device_interface());
//    std::cerr << "Allocating buff_bx on GPU" << std::endl;
    //    buff_bx.device_malloc(halide_cuda_device_interface());
    //   std::cerr << "Allocating output on GPU" << std::endl;
    //    buff_output.device_malloc(halide_cuda_device_interface());
#endif
#else
    Halide::Buffer<C_DATA_TYPE> buff_bx = Halide::Buffer<C_DATA_TYPE>(COLS, rows_per_proc);   
#endif
#ifdef CHECK_RESULTS
    std::cerr << "Filling buff_input"  << std::endl;
    int next = 0;
    for (int y = 0; y < rows_per_proc; y++) {
        for (int x = 0; x < COLS; x++) {
          buff_input(x,y) = (C_DATA_TYPE)(next++ % 1000);
        }
    }
    if (rank < NODES && NODES != 1) { // need to fill up the last two rows with the first two rows of next rank on the machine. We'll just assume we can algorithmically generate it here
      next = 0;
      for (int y = rows_per_proc; y < rows_per_proc + 2; y++) {
        for (int x = 0; x < COLS; x++) {
          //          buff_input(x,y) = next++;
          buff_input(x,y) = (C_DATA_TYPE)(next++ % 1000);
        }
      }      
    }
#endif // otherwise, don't really care about the actual values b/c we aren't concerned with the filling time
#ifdef DISTRIBUTE
#else

    //    Halide::Buffer<C_DATA_TYPE> buff_output = Halide::Buffer<C_DATA_TYPE>(COLS - 2, rows_per_proc - 2);
    Halide::Buffer<C_DATA_TYPE> buff_output = Halide::Buffer<C_DATA_TYPE>(COLS, rows_per_proc);
#endif
#ifdef CPU_ONLY
    std::cerr << "Running once for warm up"  << std::endl;
    blur_dist(buff_input.raw_buffer(), buff_bx.raw_buffer(), buff_output.raw_buffer());
#elif defined(GPU_ONLY)
    std::cerr << "Running once for warm up"  << std::endl;
    blur_dist_gpu(buff_input.raw_buffer()/*, buff_input.raw_buffer(), buff_bx.raw_buffer(), buff_output.raw_buffer()*/, buff_output.raw_buffer());//, buff_wait.raw_buffer());
    //    buff_output.raw_buffer()->set_device_dirty(false);
#endif
    
#ifdef DISTRIBUTE
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPU_ONLY
    assert(cuCtxSynchronize() == 0);
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
        blur_dist(buff_input.raw_buffer(), buff_bx.raw_buffer(), buff_output.raw_buffer());
#elif defined(GPU_ONLY)        
        blur_dist_gpu(buff_input.raw_buffer(), /*buff_input.raw_buffer(), buff_bx.raw_buffer(), buff_output.raw_buffer(),*/ buff_output.raw_buffer());//, buff_wait.raw_buffer());
        //        buff_output.raw_buffer()->set_device_dirty(false);
#endif
#ifdef DISTRIBUTE
        MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPU_ONLY
    assert(cuCtxSynchronize() == 0);        
#endif
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            std::chrono::duration<double,std::milli> duration = end - start;
            duration_vector.push_back(duration);
            std::cerr << "Iteration " << i << " done in " << duration.count() << "ms." << std::endl;
        }
#if defined(CHECK_RESULTS) && defined(DISTRIBUTE)
        if (i == 0) {
            std::string output_fn = "./build/blur_dist_rank_" + std::to_string(rank) + ".txt";
            std::ofstream myfile;
            myfile.open(output_fn);
            for (int y = 0; y < /*((rank == PROCS - 1) ? */(rows_per_proc - 2) /*: rows_per_proc)*/; y++) {
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
#ifdef GPU_ONLY
        //    cudaDeviceSynchronize();
        std::cerr << "Calling synchronize" << std::endl;
        assert(cuCtxSynchronize() == 0);
#endif
    }
    
    if (rank == 0) {
      print_time("performance_CPU.csv", "blur_dist", {"Tiramisu_dist"}, {median(duration_vector)});
        std::cout.flush();
        
#if defined(CHECK_RESULTS) && defined(DISTRIBUTE)
 // combine the rank files together
        C_DATA_TYPE *got = (C_DATA_TYPE*)malloc(sizeof(C_DATA_TYPE) * (ROWS - 2) * (COLS - 2));
        int idx = 0;
        for (int n = 0; n < PROCS; n++) {
            std::ifstream in_file;
            in_file.open("./build/blur_dist_rank_" + std::to_string(n) + ".txt");
            std::string line;
            while(std::getline(in_file, line)) {
                got[idx++] = (C_DATA_TYPE)std::stof(line);
            }
            in_file.close();
        }
        next = 0;
        Halide::Buffer<C_DATA_TYPE> full_input = Halide::Buffer<C_DATA_TYPE>(COLS, ROWS);
        for (int y = 0; y < ROWS; y++) {
          //            if (y % rows_per_proc == 0) {
          //              next = 0;
          //            }
            for (int x = 0; x < COLS; x++) {
              full_input(x, y) = (next++ % 1000);
            }
        }
        idx = 0;
	std::cerr << "Comparing" << std::endl;
        for (int r = 0; r < ROWS - 2; r++) {
            for (int c = 0; c < COLS - 2; c++) {
              C_DATA_TYPE should_be = std::floor((C_DATA_TYPE)((full_input(c,r) + full_input(c+1, r) + full_input(c+2, r) + full_input(c, r+1) +
                                                      full_input(c, r+2) + full_input(c+1, r+1) + full_input(c+1, r+2) + full_input(c+2, r+1) +
                                                                full_input(c+2, r+2)) / (C_DATA_TYPE)9));
              C_DATA_TYPE is = std::floor(got[idx++]);
              if (std::fabs(should_be - is) > 0.0f) {
                std::cerr << "Mismatch at row " << r << " column " << c << ". Should be " << should_be << ", but is " << is << std::endl;
                assert(false);
              }
            }
        }
        free(got);

#elif defined(CHECK_RESULTS) // not distributed
	std::cerr << "Comparing" << std::endl;
        next = 0;
        Halide::Buffer<C_DATA_TYPE> full_input = Halide::Buffer<C_DATA_TYPE>(COLS, ROWS);
        for (int y = 0; y < ROWS; y++) {
            for (int x = 0; x < COLS; x++) {
              //                full_input(x, y) = next++;
              full_input(x,y) = (C_DATA_TYPE)(next++ % 1000);
            }
        }
        for (int r = 0; r < ROWS - 2; r++) {
          for (int c = 0; c < COLS - 2; c++) {
            C_DATA_TYPE should_be = (C_DATA_TYPE)std::floor((C_DATA_TYPE)((full_input(c,r) + full_input(c+1, r) + full_input(c+2, r) + full_input(c, r+1) +
                                                  full_input(c, r+2) + full_input(c+1, r+1) + full_input(c+1, r+2) + full_input(c+2, r+1) +
                                                                           full_input(c+2, r+2)) / (C_DATA_TYPE)9));
            C_DATA_TYPE is = (C_DATA_TYPE)std::floor(buff_output(c,r));
            if(std::fabs(should_be - is) > (C_DATA_TYPE)0) {
              //              fprintf(stderr, "%g, %g\n", should_be, buff_output(c,r));
              //              std::cerr << std::fabs(should_be - is) << std::endl;
              std::cerr << "is: " << is << std::endl;
              std::cerr << "should be : " << should_be << std::endl;
              std::cerr << "At halide point: (" << c << ", " << r << ")" << std::endl;
              assert(false);
            }
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
