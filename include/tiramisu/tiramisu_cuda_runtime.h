//
// Created by Jessica Ray on 1/9/18.
//

#ifndef TIRAMISU_TIRAMISU_CUDA_RUNTIME_H
#define TIRAMISU_TIRAMISU_CUDA_RUNTIME_H

#include "cuda.h"

extern "C" {
struct cuda_vars {
    CUdevice device;
    CUcontext ctx;
    CUmodule mod1;
    CUmodule mod2;
    CUmodule mod3;
    CUmodule mod4;
    CUmodule mod5;
};

extern struct cuda_vars cvars;
}

#endif //TIRAMISU_TIRAMISU_CUDA_RUNTIME_H
