//
// Created by Jessica Ray on 12/6/17.
//

#ifndef TIRAMISU_TIRAMISU_CUDA_H
#define TIRAMISU_TIRAMISU_CUDA_H

#include "HalideRuntime.h"

// overrides of the weak halide functions for CUDA functions

int halide_device_malloc(void *user_context, struct halide_buffer_t *buf,
                         const halide_device_interface_t *device_interface);

int halide_device_free(void *user_context, struct halide_buffer_t *buf);

int halide_copy_to_host(void *user_context, struct halide_buffer_t *buf);

int halide_copy_to_device(void *user_context, struct halide_buffer_t *buf,
                          const struct halide_device_interface_t *device_interface);

#endif //TIRAMISU_TIRAMISU_CUDA_H
