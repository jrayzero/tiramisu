//
// Created by Jessica Ray on 11/28/17.
//

#ifndef TIRAMISU_WRAPPER_BLUR_H
#define TIRAMISU_WRAPPER_BLUR_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int blur_dist(halide_buffer_t *, halide_buffer_t *);
int blur_dist_argv(void **args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_BLUR_H
