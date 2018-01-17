//
// Created by Jessica Ray on 11/27/17.
//

#ifndef TIRAMISU_BLUR_PARAMS_H
#define TIRAMISU_BLUR_PARAMS_H

//#define CPU_ONLY
//#define GPU_ONLY
#define COOP
#define DISTRIBUTE
#define PARALLEL

#ifdef COOP
#define GPU_PERCENTAGE 8
#endif

//#define CHECK_RESULTS
 
#define REQ MPI_THREAD_FUNNELED
#define PRINT_ITER_0
#ifdef CHECK_RESULTS
#define ITERS 1
#else
#define ITERS 10
#endif

#ifdef CHECK_RESULTS
#define ROWS 20000//4000
#define COLS 480//0
#else
#define ROWS 16000
#define COLS 4500000 
#endif

#define T_LOOP_ITER_TYPE tiramisu::p_int64
#define C_LOOP_ITER_TYPE int64_t
#define T_DATA_TYPE tiramisu::p_float32
#define C_DATA_TYPE float


#endif //TIRAMISU_BLUR_PARAMS_H
