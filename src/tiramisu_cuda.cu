//
// Created by Jessica Ray on 12/6/17.
//

#include <cassert>
#include "tiramisu/tiramisu_cuda.h"

void tiramisu_cuda_malloc(void *device_ptr, size_t bytes) {
    assert(cudaMalloc(&device_ptr, bytes) == 0 && "tiramisu_cuda_malloc failed");
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