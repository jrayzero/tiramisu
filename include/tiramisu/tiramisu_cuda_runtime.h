//
// Created by Jessica Ray on 1/9/18.
//

#ifndef TIRAMISU_TIRAMISU_CUDA_RUNTIME_H
#define TIRAMISU_TIRAMISU_CUDA_RUNTIME_H

#include "cuda.h"

struct cuda_vars {
    CUdevice device;
    CUcontext ctx;
};

#endif //TIRAMISU_TIRAMISU_CUDA_RUNTIME_H
