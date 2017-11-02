//
// Created by Jessica Ray on 10/17/17.
//

#ifndef TIRAMISU_WRAPPER_DTEST_04_H
#define TIRAMISU_WRAPPER_DTEST_04_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int dtest_04(halide_buffer_t *_b_input_buffer, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *_b_blury_buffer);
int dtest_04_argv(void **args);
// Result is never null and points to constant static data
const struct halide_filter_metadata_t *dtest_04_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif
#endif //TIRAMISU_WRAPPER_DTEST_04_H
