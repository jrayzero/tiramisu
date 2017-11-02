//
// Created by Jessica Ray on 10/17/17.
//

#ifndef TIRAMISU_WRAPPER_DTEST_05_H
#define TIRAMISU_WRAPPER_DTEST_05_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int dtest_05(halide_buffer_t *_b_input_buffer, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *_b_blury_buffer);
int dtest_05_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *dtest_05_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif //TIRAMISU_WRAPPER_DTEST_05_H
