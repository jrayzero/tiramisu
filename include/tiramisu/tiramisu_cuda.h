//
// Created by Jessica Ray on 12/6/17.
//

#ifndef TIRAMISU_TIRAMISU_CUDA_H
#define TIRAMISU_TIRAMISU_CUDA_H

#include "HalideRuntime.h"
#include "cuda_runtime_api.h"
#include <assert.h>
#include <stdbool.h>
#define RUNTIME
// TODO make a tiramisu_set_gpu_device to match the halide version

#ifdef DRIVER
/*extern "C" {

void tiramisu_cuda_malloc(CUdeviceptr *device_ptr, size_t bytes);

void tiramisu_cuda_free(CUdeviceptr device_ptr);

void _tiramisu_cuda_memcpy_h2d(CUdeviceptr dst, const void *src, size_t count);

void tiramisu_cuda_memcpy_d2h(void *dst, CUdeviceptr src, size_t count);

void htiramisu_cuda_malloc(halide_buffer_t *buff, size_t bytes);

void htiramisu_cuda_free(halide_buffer_t *buff);

void htiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count);

void htiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count);

}*/
#endif

#ifdef RUNTIME
extern "C" {

struct stream_tracker {

    cudaStream_t *comm_streams;

    bool *init_comm_streams;

     cudaStream_t *kernel_stream;

    bool init_kernel_stream = false;

    bool initialized = false;

  void *cuda_module;

  char *nvvm_fname;

  //  char *nvvm_data;
};

static stream_tracker st;

  void tiramisu_init_stream_tracker(int max_streams, char *name);

void tiramisu_cleanup_stream_tracker(int max_streams);

void tiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset);

void tiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count, size_t src_offset);

void tiramisu_cuda_memcpy_h2d_async(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset, int stream_id,
                                    void *buff);

void tiramisu_cuda_memcpy_d2h_async(void *dst, halide_buffer_t *src, size_t count, size_t src_offset, int stream_id,
                                    void *buff);

void tiramisu_cuda_stream_wait_event(void *buff, int stream_id);

void _tiramisu_cuda_memcpy_h2d(void *dst, const void *src, size_t count);

void _tiramisu_cuda_memcpy_d2h(void *dst, const void *src, size_t count);

void _tiramisu_cuda_memcpy_h2d_async(void *dst, const void *src, size_t count, int stream_id, void *buff);

void _tiramisu_cuda_memcpy_d2h_async(void *dst, const void *src, size_t count, int stream_id, void *buff);

void tiramisu_cuda_malloc(void **device_ptr, size_t bytes);

void tiramisu_cuda_free(void *device_ptr);

cudaStream_t tiramisu_cuda_stream_create();

void tiramisu_cuda_stream_destroy(cudaStream_t stream);

void _tiramisu_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event);

cudaEvent_t tiramisu_cuda_event_create();

void tiramisu_cuda_event_destroy(cudaEvent_t event);

void tiramisu_cuda_event_record(cudaEvent_t event, cudaStream_t stream);

  /*int halide_cuda_run(void *user_context,
                    void *state_ptr,
                    const char* entry_name,
                    int blocksX, int blocksY, int blocksZ,
                    int threadsX, int threadsY, int threadsZ,
                    int shared_mem_bytes,
                    size_t arg_sizes[],
                    void* args[],
                    int8_t arg_is_buffer[],
                    int num_attributes,
                    float* vertex_buffer,
                    int num_coords_dim0,
                    int num_coords_dim1);*/

}
#endif

#endif //TIRAMISU_TIRAMISU_CUDA_H
