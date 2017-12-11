//
// Created by Jessica Ray on 12/6/17.
//

#ifndef TIRAMISU_TIRAMISU_CUDA_H
#define TIRAMISU_TIRAMISU_CUDA_H

#include "cuda_runtime_api.h"

extern "C" {
void tiramisu_cuda_malloc(void **device_ptr, size_t bytes);

void tiramisu_cuda_free(void *device_ptr);

void tiramisu_cuda_memcpy_h2d(void *dst, const void *src, size_t count);

void tiramisu_cuda_memcpy_h2h(void *dst, const void *src, size_t count);

void tiramisu_cuda_memcpy_d2h(void *dst, const void *src, size_t count);

void tiramisu_cuda_memcpy_d2d(void *dst, const void *src, size_t count);

void tiramisu_cuda_memcpy_h2d_async(void *dst, const void *src, size_t count, cudaStream_t stream);

void tiramisu_cuda_memcpy_h2h_async(void *dst, const void *src, size_t count, cudaStream_t stream);

void tiramisu_cuda_memcpy_d2h_async(void *dst, const void *src, size_t count, cudaStream_t stream);

void tiramisu_cuda_memcpy_d2d_async(void *dst, const void *src, size_t count, cudaStream_t stream);
}

#endif //TIRAMISU_TIRAMISU_CUDA_H
