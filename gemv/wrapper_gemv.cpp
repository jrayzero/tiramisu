//
// Created by Jessica Ray on 1/18/18.
//

#include "wrapper_gemv.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "tiramisu/tiramisu_cuda.h"
#include "gemv.h"
#include <cuda.h>
#include "Halide.h"
#include <math.h>
#ifdef GPU
#include "/tmp/tiramisu_cuda_runtime.h"
//#include "/tmp/tiramisu_CUDA_kernel_multiply.cu.h"
//#include "/tmp/tiramisu_CUDA_kernel_sum.cu.h"
#include "/tmp/tiramisu_CUDA_kernel_gemv.cu.h"
#endif

extern void clear_static_var_tiramisu_CUDA_kernel_gemv();
//extern void clear_static_var_tiramisu_CUDA_kernel_multiply();
//extern void clear_static_var_tiramisu_CUDA_kernel_sum();

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
        f += 0.001f;
    }
}

void fill_matrix(Halide::Buffer<float> &matrix) {
    std::cerr << "Filling matrix" << std::endl;
    float f = 0.0f;
    for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t c = 0; c < COLS; c++) {
            matrix(c,r) = f;
            f += 0.001f;
        }
        f = 0.0f;
    }
}

halide_buffer_t *fill_vector_pinned(bool fill) {
    std::cerr << "Filling vector" << std::endl;
    float *buff;
    cuMemHostAlloc((void**)&buff, COLS * sizeof(float), CU_MEMHOSTALLOC_PORTABLE);
    if (fill) {
      float f = 0.0f;
      for (uint64_t c = 0; c < COLS; c++) {
        buff[c] = f;
        f += 0.001f;
      }
    }
    halide_buffer_t *hbuff = new halide_buffer_t();
    hbuff->host = (uint8_t*)buff;
    return hbuff;
}

halide_buffer_t *fill_matrix_pinned(bool fill) {

    float *buff;
    if (fill) {
      std::cerr << "Filling matrix fully" << std::endl;
      cuMemHostAlloc((void**)&buff, COLS * ROWS * sizeof(float), CU_MEMHOSTALLOC_PORTABLE);
    } else {
      cuMemHostAlloc((void**)&buff, COLS * ROWS * sizeof(float) / 10, CU_MEMHOSTALLOC_PORTABLE);
    }
    float f = 0.0f;
    if (fill) {
      for (uint64_t r = 0; r < ROWS; r++) {
        for (uint64_t c = 0; c < COLS; c++) {
          buff[r*COLS+c] = f;
          f += 0.001f;
        }
        f = 0.0f;
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
        if (fabs(should_be[r] - result(r)) > 0.001) {
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
          should_be[r] += ((float*)(matrix->host))[r*COLS+c] * ((float*)(vector->host))[c];
        }
    }
    for (uint64_t r = 0; r < ROWS; r++) {
      if (fabs(should_be[r] - ((float*)(result->host))[r]) > 0.001f) {
            std::cerr << "result at " << r << " is wrong" << std::endl;
            std::cerr << "should be " << should_be[r] << " but is " << ((float*)(result->host))[r] << std::endl;
            std::cerr << (fabs(should_be[r] - ((float*)(result->host))[r])) << std::endl;
            assert(false);
        }
    }
    std::cerr << "It's correct!" << std::endl;
    free(should_be);
}

void run_gemv_cpu_only() {
#if defined(CPU) && !defined(FWD_PASS)
    int rank = mpi_init();
    assert(rank == 0 && "This CPU implementation is for a single node ONLY (i.e. one process)");
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    Halide::Buffer<float> vector(COLS);
    Halide::Buffer<float> matrix(COLS, ROWS);
    Halide::Buffer<float> result(ROWS);
#ifdef CHECK_RESULTS
    fill_vector(vector);
    fill_matrix(matrix);
#endif
    for (int iter = 0; iter < ITERS; iter++) {
        std::cerr << "Iter " << iter << std::endl;
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
#endif
}

void fill_weights(size_t rows, size_t cols, halide_buffer_t *buff, float starting_val) {
  float val = starting_val;
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      buff->host[r * cols + c] = val;
      val += 0.001f;
    }
  }
}

void check_fwd_pass_results(halide_buffer_t *input, std::vector<halide_buffer_t *> weights, halide_buffer_t *output) {
  // generate the correct results
  float *layer_0_res = (float*)malloc(sizeof(float)*WEIGHTS_0);
  float *layer_1_res = (float*)malloc(sizeof(float)*WEIGHTS_1);
  float *layer_2_res = (float*)malloc(sizeof(float)*WEIGHTS_2);
  float *layer_3_res = (float*)malloc(sizeof(float)*WEIGHTS_3*ROWS);

  for (size_t z = 0; z < ROWS; z++) { // total input rows
    float *row_input = &(((float*)(input->host))[z * COLS]);
    float *weights_0_1 = (float*)(weights[0]->host);
    float *weights_1_2 = (float*)(weights[1]->host);
    float *weights_2_3 = (float*)(weights[2]->host);
    float *weights_3_4 = (float*)(weights[3]->host);
    // layer 0->1
    for (size_t r = 0; r < WEIGHTS_0; r++) {
      layer_0_res[r] = 0.0f;
      for (size_t c = 0; c < COLS; c++) {
        layer_0_res[r] += weights_0_1[r * COLS + c] * row_input[c];
      }
      layer_0_res[r] = 1.0f / (1.0f + std::exp(-1.0f * layer_0_res[r]));
    }
    // layer 1->2
    for (size_t r = 0; r < WEIGHTS_1; r++) {
      layer_1_res[r] = 0.0f;
      for (size_t c = 0; c < WEIGHTS_0; c++) {
        layer_1_res[r] += weights_1_2[r * WEIGHTS_0 + c] * layer_0_res[c];
      }
      layer_1_res[r] = 1.0f / (1.0f + std::exp(-1.0f * layer_1_res[r]));
    }
    // layer 2->3
    for (size_t r = 0; r < WEIGHTS_2; r++) {
      layer_2_res[r] = 0.0f;
      for (size_t c = 0; c < WEIGHTS_1; c++) {
        layer_2_res[r] += weights_2_3[r * WEIGHTS_1 + c] * layer_1_res[c];
      }
      layer_2_res[r] = 1.0f / (1.0f + std::exp(-1.0f * layer_2_res[r]));
    }
    // layer 3->4
    for (size_t r = 0; r < WEIGHTS_3; r++) {
      layer_3_res[z * WEIGHTS_3 + r] = 0.0f;
      for (size_t c = 0; c < WEIGHTS_2; c++) {
        layer_3_res[z * WEIGHTS_3 + r] += weights_3_4[r * WEIGHTS_2 + c] * layer_2_res[c];
      }
    }
    float denom = 0.0f;
    for (size_t d = 0; d < WEIGHTS_3; d++) {
      denom += std::exp(layer_3_res[z * WEIGHTS_3 + d]);
    }
    for (size_t d = 0; d < WEIGHTS_3; d++) {
      layer_3_res[z * WEIGHTS_3 + d] = std::exp(layer_3_res[z * WEIGHTS_3 + d]) / denom;
    }
  }
  for (size_t z = 0; z < ROWS; z++) {
    float *guesses = &(((float*)(output->host))[z * WEIGHTS_3]);
    float *correct_vals = &layer_3_res[z * WEIGHTS_3];
    for (size_t c = 0; c < WEIGHTS_3; c++) {
      float guess = guesses[c];
      float correct = correct_vals[c];
      if (std::fabs(guess - correct) > 0.0001f) {
        std::cerr << "result at row " << z << ", cols << " << c << " is wrong" << std::endl;
        std::cerr << "should be " << correct << " but is " << guess << std::endl;
        assert(false);
      }
    }
  }
  std::cerr << "Passes" << std::endl;
}

void run_cpu_fwd_pass() {
#if defined(CPU) && defined(FWD_PASS)
    int rank = mpi_init();
    std::cerr << "Running cpu fwd pass" << std::endl;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    Halide::Buffer<float> input_matrix(COLS, ROWS);
    Halide::Buffer<float> weights_0_1(COLS, WEIGHTS_0);
    Halide::Buffer<float> weights_1_2(WEIGHTS_0, WEIGHTS_1);
    Halide::Buffer<float> weights_2_3(WEIGHTS_1, WEIGHTS_2);
    Halide::Buffer<float> weights_3_4(WEIGHTS_2, WEIGHTS_3);
    Halide::Buffer<float> fwd_pass_output(WEIGHTS_3,ROWS); // one per row
    fill_weights(ROWS, COLS, input_matrix.raw_buffer(), 0.0f);
    fill_weights(WEIGHTS_0, COLS, weights_0_1.raw_buffer(), 1.0f);
    fill_weights(WEIGHTS_1, WEIGHTS_0, weights_1_2.raw_buffer(), 2.0f);
    fill_weights(WEIGHTS_2, WEIGHTS_1, weights_2_3.raw_buffer(), 3.0f);
    fill_weights(WEIGHTS_3, WEIGHTS_2, weights_3_4.raw_buffer(), 4.0f);
    for (int iter = 0; iter < ITERS; iter++) {
        std::cerr << "Iter " << iter << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        gemv_cpu_fwd(input_matrix.raw_buffer(), weights_0_1.raw_buffer(), weights_1_2.raw_buffer(), weights_2_3.raw_buffer(), weights_3_4.raw_buffer(), fwd_pass_output.raw_buffer());
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end - start;
	duration_vector.push_back(duration);
	std::cerr << "Iteration " << iter << " done in " << duration.count() << "ms." << std::endl;
#ifdef CHECK_RESULTS
	if (iter == 0) {
	  check_fwd_pass_results(input_matrix.raw_buffer(), {weights_0_1.raw_buffer(), weights_1_2.raw_buffer(), weights_2_3.raw_buffer(), weights_3_4.raw_buffer()}, fwd_pass_output.raw_buffer());
	}
#endif
    }
    print_time("performance_CPU.csv", "GEMV CPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
#endif
}

void run_gemv_gpu_only() {
#ifdef GPU
    int rank = mpi_init();
    assert(rank == 0 && "This GPU implementation is for a single node ONLY (i.e. one process)");
    std::cerr << "Running GPU" << std::endl;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector;
    halide_buffer_t zeros;
    float *_zeros = (float*)calloc(ROWS, sizeof(float));
    zeros.host = (uint8_t*)_zeros;
    for (int iter = 0; iter < ITERS; iter++) {
        tiramisu_init_cuda(0);
#ifdef CHECK_RESULTS
        halide_buffer_t *vector = fill_vector_pinned(true);
        halide_buffer_t *matrix = fill_matrix_pinned(true);
#else
        halide_buffer_t *vector = fill_vector_pinned(false);
        halide_buffer_t *matrix = fill_matrix_pinned(false);
#endif
        float *res;
        assert(cuMemHostAlloc((void**)&res, ROWS * sizeof(float), CU_MEMHOSTALLOC_PORTABLE) == 0);
        halide_buffer_t result;
        result.host = (uint8_t*)res;
        std::cerr << "Iter " << iter << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        gemv_gpu(vector, matrix, &result, &zeros);
        auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> duration = end - start;
	duration_vector.push_back(duration);
	std::cerr << "Iteration " << iter << " done in " << duration.count() << "ms." << std::endl;
#ifdef CHECK_RESULTS
	if (iter == 0) {
	  check_results(vector, matrix, &result);
	}
#endif
        clear_static_var_tiramisu_CUDA_kernel_gemv();
//        clear_static_var_tiramisu_CUDA_kernel_multiply();
//        clear_static_var_tiramisu_CUDA_kernel_sum();
        cuCtxSynchronize();
        cuCtxDestroy(cvars.ctx);
    }
    print_time("performance_CPU.csv", "GEMV GPU", {"Tiramisu"}, {median(duration_vector)});
    std::cout.flush();
#endif
}


int main() {
#ifdef CPU
#ifdef FWD_PASS
  run_cpu_fwd_pass();
#else
  run_gemv_cpu_only();
#endif
#elif defined(GPU)
  run_gemv_gpu_only();
#endif
  MPI_Finalize();
}
