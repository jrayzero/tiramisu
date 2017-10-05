//
// Created by Jessica Ray on 10/3/17.
//

#ifndef TIRAMISU_WRAPPER_DTEST_01_H
#define TIRAMISU_WRAPPER_DTEST_01_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

int dtest01(halide_buffer_t *, halide_buffer_t *);
int dtest01_argv(void **args);

const struct halide_filter_metadata_t dtest01_metadata();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_DTEST_01_H