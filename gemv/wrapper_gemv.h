//
// Created by Jessica Ray on 1/18/18.
//

#ifndef TIRAMISU_WRAPPER_GEMV_H
#define TIRAMISU_WRAPPER_GEMV_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif
  // 3 layers
  int gemv_cpu_fwd(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);
  int gemv_gpu_fwd(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);
  int gemv_cpu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);
  int gemv_gpu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif //TIRAMISU_WRAPPER_GEMV_H
