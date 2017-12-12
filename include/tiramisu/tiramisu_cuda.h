//
// Created by Jessica Ray on 12/6/17.
//

#ifndef TIRAMISU_TIRAMISU_CUDA_H
#define TIRAMISU_TIRAMISU_CUDA_H

#include "HalideRuntime.h"
#include "cuda_runtime_api.h"
#define RUNTIME
// TODO make a tiramisu_set_gpu_device to match the halide version

#ifdef DRIVER
/*extern "C" {

void tiramisu_cuda_malloc(CUdeviceptr *device_ptr, size_t bytes);

void tiramisu_cuda_free(CUdeviceptr device_ptr);

void tiramisu_cuda_memcpy_h2d(CUdeviceptr dst, const void *src, size_t count);

void tiramisu_cuda_memcpy_d2h(void *dst, CUdeviceptr src, size_t count);

void htiramisu_cuda_malloc(halide_buffer_t *buff, size_t bytes);

void htiramisu_cuda_free(halide_buffer_t *buff);

void htiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count);

void htiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count);

}*/
#endif

#ifdef RUNTIME
extern "C" {

  void htiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset);

  void htiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count, size_t src_offset);

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
#endif

#endif //TIRAMISU_TIRAMISU_CUDA_H
