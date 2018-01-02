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

void tiramisu_init_stream_tracker(int max_streams) {
    assert(!st.initialized);
    st.streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * max_streams);
    st.init_streams = (bool *)calloc(max_streams, sizeof(bool)); // initialize to false
    st.initialized = true;
}

void tiramisu_cleanup_stream_tracker() {
    assert(st.initialized);
    free(st.streams);
    free(st.init_streams);
    st.initialized = false;
}

void tiramisu_cuda_malloc(void **device_ptr, size_t bytes) {
    assert(cudaMalloc(device_ptr, bytes) == 0 && "tiramisu_cuda_malloc failed");
}

void tiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset) {
    _tiramisu_cuda_memcpy_h2d(&(((float*)(dst->device))[dst_offset]), src, count);
}

void tiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count, size_t src_offset) {
    _tiramisu_cuda_memcpy_d2h(dst, &(((float*)(src->device))[src_offset]), count);
}

void tiramisu_cuda_memcpy_h2d_async(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset, int stream_id,
                                    void *buff) {
    _tiramisu_cuda_memcpy_h2d_async(&(((float*)(dst->device))[dst_offset]), src, count, stream_id, buff);
}

void tiramisu_cuda_memcpy_d2h_async(void *dst, halide_buffer_t *src, size_t count, size_t src_offset,
                                    int stream_id, void *buff) {
    _tiramisu_cuda_memcpy_d2h_async(dst, &(((float*)(src->device))[src_offset]), count, stream_id, buff);
}

void tiramisu_cuda_free(void *device_ptr) {
    assert(cudaFree(device_ptr) == 0 && "tiramisu_cuda_free failed");
}

void _tiramisu_cuda_memcpy_h2d(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) == 0 && "tiramisu_cuda_memcpy_h2d failed");
}

void _tiramisu_cuda_memcpy_h2h(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyHostToHost) == 0 && "tiramisu_cuda_memcpy_h2h failed");
}

void _tiramisu_cuda_memcpy_d2h(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) == 0 && "tiramisu_cuda_memcpy_d2h failed");
}

void _tiramisu_cuda_memcpy_d2d(void *dst, const void *src, size_t count) {
    assert(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice) == 0 && "tiramisu_cuda_memcpy_d2d failed");
}

void _tiramisu_cuda_memcpy_h2d_async(void *dst, const void *src, size_t count, int stream_id, void *buff) {
    if (!st.init_streams[stream_id]) {
        st.streams[stream_id] = tiramisu_cuda_stream_create();
        st.init_streams[stream_id] = true;
    }
    cudaStream_t stream = st.streams[stream_id];
    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream) == 0 && "tiramisu_cuda_memcpy_h2d_async failed");
    // create the cuda event
    cudaEvent_t event = tiramisu_cuda_event_create();
    tiramisu_cuda_event_record(event, stream);
    ((cudaEvent_t*)buff)[0] = event;    
}

void _tiramisu_cuda_memcpy_d2h_async(void *dst, const void *src, size_t count, int stream_id, void *buff) {
    if (!st.init_streams[stream_id]) {
        st.streams[stream_id] = tiramisu_cuda_stream_create();
        st.init_streams[stream_id] = true;
    }
    cudaStream_t stream = st.streams[stream_id];
    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream) == 0 && "tiramisu_cuda_memcpy_d2h_async failed");
    // create the cuda event
    cudaEvent_t event = tiramisu_cuda_event_create();
    tiramisu_cuda_event_record(event, stream);
    ((cudaEvent_t*)buff)[0] = event;
}

cudaStream_t tiramisu_cuda_stream_create() {
    cudaStream_t stream;
    assert(cudaStreamCreate(&stream) == 0 && "tiramisu_cuda_stream_create failed");
    return stream;
}

void tiramisu_cuda_stream_destroy(cudaStream_t stream) {
    assert(cudaStreamDestroy(stream) == 0 && "tiramisu_cuda_stream_destroy failed");
}

void _tiramisu_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
    assert(cudaStreamWaitEvent(stream, event, 0) == 0 && "tiramisu_cuda_stream_wait_event failed");
}

cudaEvent_t tiramisu_cuda_event_create() {
    cudaEvent_t event;
    assert(cudaEventCreate(&event) == 0 && "tiramisu_cuda_event_create failed");
    return event;
}

void tiramisu_cuda_event_destroy(cudaEvent_t event) {
    assert(cudaEventDestroy(event) == 0 && "tiramisu_cuda_event_destroy failed");
}

void tiramisu_cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
    assert(cudaEventRecord(event, stream) == 0 && "tiramisu_cuda_event_record failed");
    assert(cudaEventSynchronize(event) == 0 && "tiramisu_cuda_event_record synchronize failed");
}

void tiramisu_cuda_stream_wait_event(void *buff, int stream_id) {
  cudaStream_t stream = st.streams[stream_id];
  cudaEvent_t event = ((cudaEvent_t)buff);
  _tiramisu_cuda_stream_wait_event(stream, event);
}

}
#endif