//
// Created by Jessica Ray on 10/17/17.
//

#ifndef TIRAMISU_WRAPPER_DTEST_03_H
#define TIRAMISU_WRAPPER_DTEST_03_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int blurxy(halide_buffer_t *_b_input_buffer, halide_buffer_t *_b_blury_buffer);
int blurxy_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *blurxy_metadata();

#ifdef __cplusplus
}  // extern "C"

#endif //TIRAMISU_WRAPPER_DTEST_03_H
