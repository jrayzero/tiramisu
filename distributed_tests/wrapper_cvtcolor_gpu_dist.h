//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_CVTCOLOR_GPU_DIST_H
#define TIRAMISU_WRAPPER_CVTCOLOR_GPU_DIST_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int cvtcolor_gpu_dist(halide_buffer_t *, halide_buffer_t *);
int cvtcolor_gpu_dist_argv(void **args);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_CVTCOLOR_GPU_DIST_H
