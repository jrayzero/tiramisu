//
// Created by Jessica Ray on 12/6/17.
//

#include <cassert>
#include "tiramisu/tiramisu_cuda.h"
#include "tiramisu/tiramisu_cuda_runtime.h"
#include <stdio.h>
#include <HalideBuffer.h>
#include "HalideRuntime.h"
#include "cuda.h"


extern "C" {

// TODO load all at once into one module to reduce overhead of loading module everytime at runtime
void *tiramisu_init_cuda(int device_num) {
    struct cuda_vars *cvars = (struct cuda_vars*)malloc(sizeof(struct cuda_vars));
    assert(cuInit(0) == 0);
    assert(cuDeviceGet(&(cvars->device), device_num) == 0);
    size_t memory;
    cuDeviceTotalMem(&memory, cvars->device);
    fprintf(stderr, "Total memory on device %d is %lu\n", device_num, memory);
    assert(cuCtxCreate(&(cvars[0].ctx), 0, cvars[0].device) == 0);
    return (void*)cvars;
}

inline void tiramisu_check_cudad_error(const char *wrapper_name, CUresult code) {
    if (code != CUDA_SUCCESS) {
        const char *name, *desc;
        cuGetErrorName(code, &name);
        cuGetErrorString(code, &desc);
        fprintf(stderr, "Wrapper %s had failure %s: %s\n", wrapper_name, name, desc);
        exit(29);
    }
}

inline void _tiramisu_cudad_malloc(CUdeviceptr *device_ptr, size_t bytes) {
  tiramisu_check_cudad_error("tiramisu_cudad_malloc", cuMemAlloc(device_ptr, bytes));
}

inline void _tiramisu_cudad_free(CUdeviceptr device_ptr) {
    tiramisu_check_cudad_error("tiramisu_cudad_free", cuMemFree(device_ptr));
}

inline void _tiramisu_cudad_memcpy_h2d(CUdeviceptr dst, const void *src, size_t count) {
    tiramisu_check_cudad_error("tiramisu_cudad_memcpy_h2d", cuMemcpyHtoD(dst, src, count));
}

inline void _tiramisu_cudad_memcpy_d2h(void *dst, CUdeviceptr src, size_t count) {
    tiramisu_check_cudad_error("tiramisu_cudad_memcpy_h2d", cuMemcpyDtoH(dst, src, count));
}

inline void _tiramisu_cudad_memcpy_async_h2d(CUdeviceptr dst, const void *src, size_t count, CUstream stream, CUevent event) {
    tiramisu_check_cudad_error("tiramisu_cudad_memcpy_async_h2d", cuMemcpyHtoDAsync(dst, src, count, stream));
    tiramisu_cudad_event_record(event, stream);
}

inline void _tiramisu_cudad_memcpy_async_d2h(void *dst, CUdeviceptr src, size_t count, CUstream stream, CUevent event) {
    tiramisu_check_cudad_error("tiramisu_cudad_memcpy_async_d2h", cuMemcpyDtoHAsync(dst, src, count, stream));
    tiramisu_cudad_event_record(event, stream);
}

/*void *tiramisu_cudad_stream_create(int num_streams) {
  CUstream *streams = (CUstream*)malloc(sizeof(CUstream) * num_streams);
  for (int i = 0; i < num_streams; i++) {
      CUstream stream;
      tiramisu_check_cudad_error("tiramisu_cudad_stream_create", cuStreamCreate(&stream, 0));
      streams[i] = stream;
  }
  return (void*)streams;
  }*/

void tiramisu_cudad_stream_create(void *_stream_buff, int num_streams) {
    CUstream *stream_buff = (CUstream*)_stream_buff;
    for (int i = 0; i < num_streams; i++) {
        CUstream stream;
        tiramisu_check_cudad_error("tiramisu_cudad_stream_create", cuStreamCreate(&stream, 0));
        stream_buff[i] = stream;
    }

    //    halide_buffer_t *stream_buff = (halide_buffer_t*)malloc(sizeof(halide_buffer_t));
    //    stream_buff->host = (uint8_t*)malloc(sizeof(void*)*num_streams);
    //    for (int i = 0; i < num_streams; i++) {
    //        CUstream stream;
    //        tiramisu_check_cudad_error("tiramisu_cudad_stream_create", cuStreamCreate(&stream, 0));
    //        fprintf(stderr, "%d\n", i);
    //            stream_buff->host[0] = (uint8_t)
    //        //        stream_buff[i].host = (uint8_t*)stream;
    //        //        ((CUstream*)(stream_buff->host))[i] = stream;
    //
    //    }
    //    return stream_buff;
}

void tiramisu_cudad_stream_destroy(CUstream stream) {
    tiramisu_check_cudad_error("tiramisu_cudad_stream_destroy", cuStreamDestroy(stream));
}

void *tiramisu_cudad_event_create() {
    CUevent event;
    tiramisu_check_cudad_error("tiramisu_cudad_event_create", cuEventCreate(&event, 0));
    return (void*)event;
}

void tiramisu_cudad_event_destroy(CUevent event) {
    tiramisu_check_cudad_error("tiramisu_cudad_event_destroy", cuEventDestroy(event));
}

void tiramisu_cudad_event_record(CUevent event, CUstream stream) {
    tiramisu_check_cudad_error("tiramisu_cudad_event_record", cuEventRecord(event, stream));
}

  void tiramisu_cudad_stream_wait_event(void *stream_buff, void *event_buff) {    
    tiramisu_check_cudad_error("tiramisu_cudad_stream_wait_event", cuStreamWaitEvent(((CUstream*)stream_buff)[0], ((CUevent*)event_buff)[0], 0));
}

//void tiramisu_cudad_malloc(halide_buffer_t *buff, size_t bytes) {
//    CUdeviceptr p;
//    _tiramisu_cudad_malloc(&p, bytes);
//    buff->device = p;
//}

struct halide_buffer_t *tiramisu_cudad_malloc(size_t bytes) {
    CUdeviceptr p;
    _tiramisu_cudad_malloc(&p, bytes);
    struct halide_buffer_t *buff = (halide_buffer_t*)malloc(sizeof(struct halide_buffer_t));
    buff->device = p;
    fprintf(stderr, "gpu malloc %lu bytes at %p\n", bytes, (void*)p); 
    return buff;
}

void tiramisu_cudad_free(halide_buffer_t *buff) {
    _tiramisu_cudad_free((CUdeviceptr)(buff->device));
}

  void tiramisu_cudad_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count, size_t dst_index) {
    _tiramisu_cudad_memcpy_h2d((CUdeviceptr)(dst->device) + dst_index, src, count);
}

  void tiramisu_cudad_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count, size_t src_index) {
    _tiramisu_cudad_memcpy_d2h(dst, (CUdeviceptr)(src->device) + src_index, count);
}

  void tiramisu_cudad_memcpy_async_h2d(halide_buffer_t *dst, const void *src, size_t count, void *stream_buff, void *event_buff, size_t dst_index) {
    CUevent event;
    tiramisu_check_cudad_error("tiramisu_cudad_memcpy_async_h2d", cuEventCreate(&event, 0));
    ((CUevent*)event_buff)[0] = event;
    CUstream stream = ((CUstream*)stream_buff)[0];
    _tiramisu_cudad_memcpy_async_h2d((CUdeviceptr)(dst->device) + dst_index, src, count, stream, event);
}

//  void tiramisu_cudad_memcpy_async_h2d_buff(Halide::Runtime::Buffer<> *dst, const void *src, size_t count, void *stream, void *event_buff) {
//  _tiramisu_cudad_memcpy_async_h2d((CUdeviceptr)(dst->raw_buffer()->device), src, count, (CUstream)stream, ((CUevent*)event_buff)[0]);
//}


void tiramisu_cudad_memcpy_async_d2h(void *dst, halide_buffer_t *src, size_t count, void *stream_buff, void *event_buff, size_t src_index) {
    CUevent event;
    tiramisu_check_cudad_error("tiramisu_cudad_memcpy_async_d2h", cuEventCreate(&event, 0));
    ((CUevent*)event_buff)[0] = event;
    _tiramisu_cudad_memcpy_async_d2h(dst, (CUdeviceptr)(src->device) + src_index, count, ((CUstream*)stream_buff)[0], event);
}

}

#ifdef RUNTIME

//extern "C" {
//
//// TODO load all at once into one module to reduce overhead of loading module everytime at runtime
//void *tiramisu_init_cuda(int device_num) {
//    struct cuda_vars *cvars = (struct cuda_vars*)malloc(sizeof(struct cuda_vars));
//    assert(cuInit(0) == 0);
//    assert(cuDeviceGet(&(cvars->device), device_num) == 0);
//    size_t memory;
//    cuDeviceTotalMem(&memory, cvars->device);
//    fprintf(stderr, "Total memory on device %d is %lu\n", device_num, memory);
//    assert(cuCtxCreate(&(cvars[0].ctx), 0, cvars[0].device) == 0);
//    return (void*)cvars;
//}
//
//void *get_kernel_stream() {
//    assert(st.init_kernel_stream);
//    return *(st.kernel_stream);
//}
//
//CUstream tiramisu_get_kernel_stream() {
//    assert(st.init_kernel_stream);
//    return (CUstream)st.kernel_stream;
//}
//
//void tiramisu_init_stream_tracker(int max_streams, char *name) {
//    assert(!st.initialized);
//    st.comm_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * max_streams);
//    st.init_comm_streams = (bool *)calloc(max_streams, sizeof(bool)); // initialize to false
//    st.initialized = true;
//    //    if (!st.init_kernel_stream) {
//    st.kernel_stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
//    //      st.kernel_stream[0] = tiramisu_cuda_stream_create();
//    //      st.init_kernel_stream = true;
//    //      fprintf(stderr, "query res %d\n", cudaStreamQuery((CUstream_st*)get_kernel_stream()));
//    /*      if (err != 0) {
//      fprintf(stderr, "%d\n", err);
//    }
//    st.kernel_stream = (cudaStream_t)s;*/
//    //      fprintf(stderr, "created kernel stream %p\n", st.kernel_stream);
//    //    }
//    st.nvvm_fname = name;
//}
//
//void tiramisu_cleanup_stream_tracker(int max_streams) {
//    assert(st.initialized);
//    for (int i = 0; i < max_streams; i++) {
//        tiramisu_cuda_stream_destroy(st.comm_streams[i]);
//    }
//    tiramisu_cuda_stream_destroy(st.kernel_stream[0]);
//    free(st.comm_streams);
//    free(st.init_comm_streams);
//    free(st.kernel_stream);
//    st.initialized = false;
//    st.init_kernel_stream = false;
//    fprintf(stderr, "did stream tracker cleanup\n");
//}
//
//void tiramisu_cuda_malloc(void **device_ptr, size_t bytes) {
//    assert(cudaMalloc(device_ptr, bytes) == 0 && "tiramisu_cudad_malloc failed");
//}
//
//void tiramisu_cuda_memcpy_h2d(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset) {
//    _tiramisu_cuda_memcpy_h2d(&(((float*)(dst->device))[dst_offset]), src, count);
//}
//
//void tiramisu_cuda_memcpy_d2h(void *dst, halide_buffer_t *src, size_t count, size_t src_offset) {
//    _tiramisu_cuda_memcpy_d2h(dst, &(((float*)(src->device))[src_offset]), count);
//}
//
//void tiramisu_cuda_memcpy_h2d_async(halide_buffer_t *dst, const void *src, size_t count, size_t dst_offset, int stream_id,
//                                    void *buff) {
//    _tiramisu_cuda_memcpy_h2d_async(&(((float*)(dst->device))[dst_offset]), src, count, stream_id, buff);
//}
//
//void tiramisu_cuda_memcpy_d2h_async(void *dst, halide_buffer_t *src, size_t count, size_t src_offset,
//                                    int stream_id, void *buff) {
//    _tiramisu_cuda_memcpy_d2h_async(dst, &(((float*)(src->device))[src_offset]), count, stream_id, buff);
//}
//
//void tiramisu_cuda_free(void *device_ptr) {
//    assert(cudaFree(device_ptr) == 0 && "tiramisu_cudad_free failed");
//}
//
//void _tiramisu_cuda_memcpy_h2d(void *dst, const void *src, size_t count) {
//    assert(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) == 0 && "tiramisu_cudad_memcpy_h2d failed");
//}
//
//void _tiramisu_cuda_memcpy_h2h(void *dst, const void *src, size_t count) {
//    assert(cudaMemcpy(dst, src, count, cudaMemcpyHostToHost) == 0 && "tiramisu_cuda_memcpy_h2h failed");
//}
//
//void _tiramisu_cuda_memcpy_d2h(void *dst, const void *src, size_t count) {
//    assert(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) == 0 && "tiramisu_cudad_memcpy_d2h failed");
//}
//
//void _tiramisu_cuda_memcpy_d2d(void *dst, const void *src, size_t count) {
//    assert(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice) == 0 && "tiramisu_cuda_memcpy_d2d failed");
//}
//
//void _tiramisu_cuda_memcpy_h2d_async(void *dst, const void *src, size_t count, int stream_id, void *buff) {
//    //  fprintf(stderr, "memcpy h2d async\n");
//    if (!st.init_comm_streams[stream_id]) {
//        st.comm_streams[stream_id] = tiramisu_cuda_stream_create();
//        st.init_comm_streams[stream_id] = true;
//    }
//    cudaStream_t stream = st.comm_streams[stream_id];
//    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream) == 0 && "tiramisu_cuda_memcpy_h2d_async failed");
//    // create the cuda event
//    cudaEvent_t event = tiramisu_cuda_event_create();
//    // You record the event of a particular stream. Later on you'd have another stream wait for this event.
//    // This gives you a way to synchronize across streams
//    tiramisu_cuda_event_record(event, stream);//st.kernel_stream);//stream);
//    ((cudaEvent_t*)buff)[0] = event;
//}
//
//void _tiramisu_cuda_memcpy_d2h_async(void *dst, const void *src, size_t count, int stream_id, void *buff) {
//    if (!st.init_comm_streams[stream_id]) {
//        st.comm_streams[stream_id] = tiramisu_cuda_stream_create();
//        st.init_comm_streams[stream_id] = true;
//    }
//    cudaStream_t stream = st.comm_streams[stream_id];
//    assert(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream) == 0 && "tiramisu_cuda_memcpy_d2h_async failed");
//    // create the cuda event
//    cudaEvent_t event = tiramisu_cuda_event_create();
//    tiramisu_cuda_event_record(event, stream);//st.kernel_stream);//stream);
//    ((cudaEvent_t*)buff)[0] = event;
//}
//
//cudaStream_t tiramisu_cuda_stream_create() {
//    cudaStream_t stream;
//    assert(cudaStreamCreate(&stream) == 0 && "tiramisu_cuda_stream_create failed");
//    return stream;
//}
//
//void tiramisu_cuda_stream_destroy(cudaStream_t stream) {
//    assert(cudaStreamDestroy(stream) == 0 && "tiramisu_cuda_stream_destroy failed");
//}
//
//void _tiramisu_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
//    cudaError_t res = cudaStreamWaitEvent(stream, event, 0);
//    if (res != 0) {
//        fprintf(stderr, "%d, %p, %p\n", res, stream, event);
//        assert(false);
//    }
//    //    assert(cudaStreamWaitEvent(stream, event, 0) == 0 && "tiramisu_cuda_stream_wait_event failed");
//}
//
//cudaEvent_t tiramisu_cuda_event_create() {
//    cudaEvent_t event;
//    assert(cudaEventCreate(&event) == 0 && "tiramisu_cuda_event_create failed");
//    return event;
//}
//
//void tiramisu_cuda_event_destroy(cudaEvent_t event) {
//    assert(cudaEventDestroy(event) == 0 && "tiramisu_cuda_event_destroy failed");
//}
//
//void tiramisu_cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
//    assert(cudaEventRecord(event, stream) == 0 && "tiramisu_cuda_event_record failed");
//}
//
///*  void tiramisu_cuda_stream_wait_event(void *buff, int stream_id) {
//cudaStream_t stream = st.comm_streams[stream_id];
//cudaEvent_t event = ((cudaEvent_t)buff);
//_tiramisu_cuda_stream_wait_event(stream, event);
//}*/
//
//void tiramisu_cuda_stream_wait_event(void *buff, int stream_id) {
//    cudaEvent_t event = ((cudaEvent_t)buff);
//    assert(st.kernel_stream[0] != NULL);
//    _tiramisu_cuda_stream_wait_event(st.kernel_stream[0], event); // block the kernel stream on the communication event
//}
//
//int halide_launch_cuda_kernel(CUfunction f,int blocksX, int blocksY, int blocksZ,
//                              int threadsX, int threadsY, int threadsZ,
//                              int shared_mem_bytes, void **translated_args) {
//    if (!st.init_kernel_stream) {
//        CUstream stream = tiramisu_cuda_stream_create();//(CUstream)get_kernel_stream();//*((CUstream*)get_kernel_stream());
//        st.kernel_stream[0] = stream;
//        st.init_kernel_stream = true;
//    }
//    CUresult err = cuLaunchKernel(f,
//                                  blocksX,  blocksY,  blocksZ,
//                                  threadsX, threadsY, threadsZ,
//                                  shared_mem_bytes,
//                                  *(st.kernel_stream),
//                                  translated_args,
//                                  NULL);
//    if (err != CUDA_SUCCESS) {
//        fprintf(stderr, "%d\n", err);
//        assert(false && "cuLaunchKernel failed");
//    }
//    return 0;
//
//}


//}
#endif
