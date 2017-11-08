//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_GAUSSIAN_DIST_H
#define TIRAMISU_WRAPPER_GAUSSIAN_DIST_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

  int gaussian_dist(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);
int gaussian_dist_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *gaussian_data_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_GAUSSIAN_DIST_H
