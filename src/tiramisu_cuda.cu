//
// Created by Jessica Ray on 12/6/17.
//

#include "tiramisu/tiramisu_cuda.h"

int halide_device_malloc(void *user_context, struct halide_buffer_t *buf,
                         const halide_device_interface_t *device_interface) {
    return 0;
}

int halide_device_free(void *user_context, struct halide_buffer_t *buf) {
    return 0;
}

int halide_copy_to_host(void *user_context, struct halide_buffer_t *buf) {
    return 0;
}

int halide_copy_to_device(void *user_context, struct halide_buffer_t *buf,
                          const struct halide_device_interface_t *device_interface) {
    return 0;
}