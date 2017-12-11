//
// Created by Jessica Ray on 11/27/17.
//

#ifndef TIRAMISU_BLUR_PARAMS_H
#define TIRAMISU_BLUR_PARAMS_H

//#define CPU_ONLY
#define GPU_ONLY
//#define HYBRID
//#define GPU_LOAD_FACTOR 80 // put 80% of the data onto the GPU

#define REQ MPI_THREAD_FUNNELED
#define PRINT_ITER_0
#define ITERS 20

#define CHECK_RESULTS

#define NODES 4
#define ROWS 1000
#define COLS 100
#if defined(CPU_ONLY) || defined(HYBRID)
#define T_LOOP_ITER_TYPE tiramisu::p_int64
#define C_LOOP_ITER_TYPE int64_t
#elif defined(GPU_ONLY)
#define T_LOOP_ITER_TYPE tiramisu::p_int32
#define C_LOOP_ITER_TYPE int32_t
#endif
#define T_DATA_TYPE tiramisu::p_float32
#define C_DATA_TYPE float



#endif //TIRAMISU_BLUR_PARAMS_H