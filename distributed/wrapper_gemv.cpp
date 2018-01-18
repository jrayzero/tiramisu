//
// Created by Jessica Ray on 1/18/18.
//

#include "wrapper_gemv.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "tiramisu/tiramisu_cuda.h"
#include "gemv_params.h"
#include <cuda.h>
#include "Halide.h"
#include <math.h>
#ifdef GPU
#include "/tmp/tiramisu_cuda_runtime.h"
#include "/tmp/tiramisu_CUDA_kernel_gemv.cu.h"
#endif

int mpi_init() {
    int provided = -1;
    MPI_Init_thread(NULL, NULL, REQ, &provided);
    assert(provided == REQ && "Did not get the appropriate MPI thread requirement.");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void fill_vector(Halide::Buffer<float> &vector) {
    std::cerr << "Filling vector" << std::endl;
    float f = 0.0f;
    for (uint64_t c = 0; c < COLS; c++) {
        vector(c) = f;
        f += 0.01;
    }
}

void fill_matrix(Halide::Buffer<float> &matrix) {
    std::cerr << "Filling matrix" << std::endl;
    float f = 0.0f;
    for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t c = 0; c < COLS; c++) {
            matrix(c,r) = f;
            f += 0.01;
        }
    }
}

halide_buffer_t *fill_vector_pinned() {
    std::cerr << "Filling vector" << std::endl;
    float *buff;
    cuMemHostAlloc((void**)&buff, COLS * sizeof(float), CU_MEMHOSTALLOC_PORTABLE);
    float f = 0.0f;
    for (uint64_t c = 0; c < COLS; c++) {
      buff[c] = f;
      f += 0.01;
    }
    halide_buffer_t *hbuff = new halide_buffer_t();
    hbuff->host = (uint8_t*)buff;
    return hbuff;
}

halide_buffer_t *fill_matrix_pinned() {
    std::cerr << "Filling matrix" << std::endl;
    float *buff;
    cuMemHostAlloc((void**)&buff, COLS * ROWS * sizeof(float), CU_MEMHOSTALLOC_PORTABLE);
    float f = 0.0f;
    for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t c = 0; c < COLS; c++) {
            buff[r*COLS+c] = f;
            f += 0.01;
        }
    }
    halide_buffer_t *hbuff = new halide_buffer_t();
    hbuff->host = (uint8_t*)buff;
    return hbuff;
}

void check_results(Halide::Buffer<float> vector, Halide::Buffer<float> matrix, Halide::Buffer<float> result) {
    float *should_be = (float*)calloc(ROWS, sizeof(float));
    for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t c = 0; c < COLS; c++) {
            should_be[r] += matrix(c,r) * vector(c);
        }
    }
    for (uint64_t r = 0; r < ROWS; r++) {
        if (fabs(should_be[r] - result(r)) > 0.000001) {
            std::cerr << "result at " << r << " is wrong" << std::endl;
            std::cerr << "should be " << should_be[r] << " but is " << result(r) << std::endl;
            assert(false);
        }
    }
    std::cerr << "It's correct!" << std::endl;
    free(should_be);
}

void check_results(halide_buffer_t *vector, halide_buffer_t *matrix, halide_buffer_t *result) {
    float *should_be = (float*)calloc(ROWS, sizeof(float));
    for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t c = 0; c < COLS; c++) {
            should_be[r] += matrix[r*COLS+c] * vector[c];
        }
    }
    for (uint64_t r = 0; r < ROWS; r++) {
        if (fabs(should_be[r] - result[r]) > 0.000001) {
            std::cerr << "result at " << r << " is wrong" << std::endl;
            std::cerr << "should be " << should_be[r] << " but is " << result[r] << std::endl;
            assert(false);
        }
    }
    std::cerr << "It's correct!" << std::endl;
    free(should_be);
}

void run_gemv_cpu_only() {
    int rank = 0;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    Halide::Buffer<float> vector(COLS);
    Halide::Buffer<float> matrix(COLS, ROWS);
    Halide::Buffer<float> result(ROWS);
    fill_vector(vector);
    fill_matrix(matrix);
    for (int iter = 0; iter < ITERS; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        gemv_cpu(vector.raw_buffer(), matrix.raw_buffer(), result.raw_buffer());
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end - start;
	duration_vector.push_back(duration);
	std::cerr << "Iteration " << iter << " done in " << duration.count() << "ms." << std::endl;
#ifdef CHECK_RESULTS
	if (iter == 0) {
	  check_results(vector, matrix, result);
	}
#endif
    }
    print_time("performance_CPU.csv", "GEMV CPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
}

void run_gemv_gpu_only() {
    int rank = 0;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    halide_buffer_t *vector = fill_vector_pinned();
    halide_buffer_t *matrix = fill_matrix_pinned();
    float *res;
    assert(cuMemHostAlloc((void**)&res, ROWS * sizeof(float), CU_MEMHOSTALLOC_PORTABLE) == 0);
    halide_buffer_t result;
    result.host = res;	   
    for (int iter = 0; iter < ITERS; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        gemv_gpu(vector, matrix, &result);
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end - start;
	duration_vector.push_back(duration);
	std::cerr << "Iteration " << iter << " done in " << duration.count() << "ms." << std::endl;
#ifdef CHECK_RESULTS
	if (iter == 0) {
	  check_results(vector, matrix, &result);
	}
#endif
    }
    print_time("performance_CPU.csv", "GEMV GPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
}


int main() {

  run_gemv_cpu_only();


}
