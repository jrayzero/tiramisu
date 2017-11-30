//
// Created by Jessica Ray on 11/27/17.
//

#ifndef TIRAMISU_BLUR_PARAMS_H
#define TIRAMISU_BLUR_PARAMS_H

#define CPU_ONLY
#define REQ MPI_THREAD_FUNNELED
#define PRINT_ITER_0
#define ITERS 20

//#define DISTRIBUTE

#define ROWS 100
#define COLS 10
#define T_LOOP_ITER_TYPE tiramisu::p_int64
#define C_LOOP_ITER_TYPE int64_t
#define T_DATA_TYPE tiramisu::p_uint64
#define C_DATA_TYPE uint64_t

#endif //TIRAMISU_BLUR_PARAMS_H
