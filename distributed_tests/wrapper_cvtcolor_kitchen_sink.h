//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_CVTCOLOR_KITCHEN_SINK_H
#define TIRAMISU_WRAPPER_CVTCOLOR_KITCHEN_SINK_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

  // CPU
int cvtcolor_dist(halide_buffer_t *, halide_buffer_t *);
  // GPU
int cvtcolor_gpu_dist(halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_CVTCOLOR_KITCHEN_SINK_H
