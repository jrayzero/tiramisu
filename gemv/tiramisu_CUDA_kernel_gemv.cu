#include <cuda.h>
#include <assert.h>
#include <stdio.h>
#include "/tmp/tiramisu_cuda_runtime.h"
#include "HalideRuntime.h"

extern "C" {
__global__
void DEVICE_tiramisu_CUDA_kernel_gemv(float *result_gpu_buff, float *matrix_gpu_buff, float *vector_gpu_buff) {
  
  int block_x = blockIdx.x; // row

  int thread_x = threadIdx.x;

  __shared__ float buff[128];
  
  buff[thread_x] = 0.0f;

  for (long c7 = thread_x*8; c7 < thread_x*8+8; c7++) { // each thread does 8 elements
    buff[thread_x] += matrix_gpu_buff[block_x*1024 + c7] * vector_gpu_buff[c7];
  }

  __syncthreads();

  // now reduce the partial sums
  if (thread_x == 0) {
    for (int i = 1; i < 128; i++) {
      buff[0] += buff[i];
    }
    result_gpu_buff[block_x] = buff[0];
    //    printf("res[%d]=%f\n", block_x, result_gpu_buff[block_x]);
  }

  /*  if (thread_x < 32) {
    for (int i=32; i < 128; i+=32) {
      buff[thread_x] += buff[thread_x+i];
    }
  }
  __syncthreads();
  if (thread_x < 16) { buff[thread_x] += buff[thread_x+16]; }
  __syncthreads();
  if (thread_x < 8) { buff[thread_x] += buff[thread_x+8]; }
  __syncthreads();
  if (thread_x < 4) { buff[thread_x] += buff[thread_x+4]; }
  __syncthreads();
  if (thread_x < 2) { buff[thread_x] += buff[thread_x+2]; }
  __syncthreads();
  if (thread_x == 0) {
    result_gpu_buff[block_x] = buff[thread_x];
    }*/

  /*  for (long c7 = (long)0; c7 < ((long)7 + (long)1); c7++) {
    result_gpu_buff[((((long)0) * (1)) + (((long)block_x) * (1)))] = 
      ((matrix_gpu_buff[((((((long)8 * (long)thread_x) + (long)c7)) * (1)) + (((long)block_x) * (1024)))] * 
        vector_gpu_buff[((((((long)8 * (long)thread_x) + (long)c7)) * (1)) + 
                         (((long)0) * (1024)))]) + result_gpu_buff[((((long)0) * (1)) + (((long)block_x) * (1)))]);
                         }*/
  
}
}/*extern "C"*/

