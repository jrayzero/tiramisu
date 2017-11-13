//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_DBLURXY_DIST_H
#define TIRAMISU_WRAPPER_DBLURXY_DIST_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int dblurxy_dist(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);
int dblurxy_dist_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *dblurxy_dist_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_DBLURXY_DIST_H
