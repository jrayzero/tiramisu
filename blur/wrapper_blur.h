#ifndef TIRAMISU_WRAPPER_BLUR_H
#define TIRAMISU_WRAPPER_BLUR_H

#include <tiramisu/utils.h>

#ifdef __cplusplus
extern "C" {
#endif

  int blur_single_gpu(halide_buffer_t *, halide_buffer_t *);

  int blur_multi_gpu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

  int blur_multi_cpu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

  int blur_single_cpu(halide_buffer_t *, halide_buffer_t *, halide_buffer_t *);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //TIRAMISU_WRAPPER_BLUR_H
