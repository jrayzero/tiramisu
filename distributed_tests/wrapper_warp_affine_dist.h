//
// Created by Jessica Ray on 11/2/17.
//

#ifndef TIRAMISU_WRAPPER_WARP_AFFINE_DIST_DATA_H
#define TIRAMISU_WRAPPER_WARP_AFFINE_DIST_DATA_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int warp_affine_dist(halide_buffer_t *, halide_buffer_t *);
int warp_affine_dist_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *warp_affine_data_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_warp_affine_DIST_DATA_H
