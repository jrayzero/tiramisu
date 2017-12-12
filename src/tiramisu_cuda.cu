//
// Created by Jessica Ray on 12/6/17.
//

#include <cassert>
#include "tiramisu/tiramisu_cuda.h"
#include <stdio.h>

#ifdef DRIVER
extern "C" {

  /*void tiramisu_cuda_malloc(CUdeviceptr *device_ptr, size_t bytes) {
  assert(cuMemAlloc(device_ptr, bytes) == 0 && "tiramisu_cuda_malloc failed");
}

void tiramisu_cuda_free(CUdeviceptr device_ptr) {
    assert(cuMemFree(device_ptr) == 0 && "tiramisu_cuda_free failed");
}

  void tiramisu_cuda_memcpy_h2d(CUdeviceptr dst, const void *src, size_t count) {
    assert(cuMemcpyHtoD(dst, src, count) == 0 && "tiramisu_cuda_memcpy_h2d failed");
  }

  void tiramisu_cuda_memcpy_d2h(void *dst, CUdeviceptr src, size_t count) {
    assert(cuMemcpyDtoH(dst, src, count) == 0 && "tiramisu_cuda_memcpy_d2h failed");
  }

  void htiramisu_cuda_malloc(halide_buffer_t *buff, size_t bytes) {
    CUdeviceptr p;
    tiramisu_cuda_malloc(&p, bytes);
    buff->device = p;
  }

  void htiramisu_cuda_free(halide_buffer_t *buff) {
    tiramisu_cuda_free((CUdeviceptr)(buff->device));
  }

  void htiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count) {
    tiramisu_cuda_memcpy_h2d((CUdeviceptr)(dst->device), src, count);
  }

  void htiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count) {
    tiramisu_cuda_memcpy_d2h(dst, (CUdeviceptr)(src->device), count);
  }*/

}

#endif

#ifdef RUNTIME

extern "C" {

void tiramisu_cuda_malloc(void **device_ptr, size_t bytes) {
    assert(cudaMalloc(device_ptr, bytes) == 0 && "tiramisu_cuda_malloc failed");
}

void htiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset) {
    tiramisu_cuda_memcpy_h2d(&(((float*)(dst->device))[dst_offset]), src, count);
}

void htiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count, size_t src_offset) {
    tiramisu_cuda_memcpy_d2h(dst, &(((float*)(src->device))[src_offset]), count);
}

void tiramisu_cuda_free(void *device_ptr) {
    assert(cudaFree(device_ptr) == 0 && "tiramisu_cuda_free failed");
}

void tiramisu_cuda_memcpy_h2d(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) == 0 && "tiramisu_cuda_memcpy_h2d failed");
}

void tiramisu_cuda_memcpy_h2h(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyHostToHost) == 0 && "tiramisu_cuda_memcpy_h2h failed");
}

void tiramisu_cuda_memcpy_d2h(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) == 0 && "tiramisu_cuda_memcpy_d2h failed");
}

void tiramisu_cuda_memcpy_d2d(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice) == 0 && "tiramisu_cuda_memcpy_d2d failed");
}

void tiramisu_cuda_memcpy_h2d_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    assert(false && "Not implemented yet");
    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream) == 0 && "tiramisu_cuda_memcpy_h2d_async failed");
}

void tiramisu_cuda_memcpy_h2h_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    assert(false && "Not implemented yet");
    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, stream) == 0 && "tiramisu_cuda_memcpy_h2h_async failed");
}

void tiramisu_cuda_memcpy_d2h_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    assert(false && "Not implemented yet");
    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream) == 0 && "tiramisu_cuda_memcpy_d2h_async failed");
}

void tiramisu_cuda_memcpy_d2d_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    assert(false && "Not implemented yet");
    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream) == 0 && "tiramisu_cuda_memcpy_d2d_async failed");
}

}
#endif