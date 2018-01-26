//
// Created by Jessica Ray on 1/17/18.
//

#ifndef TIRAMISU_GEMV_PARAMS_H
#define TIRAMISU_GEMV_PARAMS_H

#define CPU
//#define GPU
#define FWD_PASS

#define CHECK_RESULTS

#ifdef FWD_PASS
#define WEIGHTS_0 1000
#define WEIGHTS_1 30
#define WEIGHTS_2 5
#define WEIGHTS_3 70
#endif

#ifndef CHECK_RESULTS
#define ROWS (int64_t)100000000
#define COLS (int64_t)500
#define ITERS 10
#else
#define ROWS (int64_t)100
#define COLS (int64_t)1024
#define ITERS 1
#endif

// types of optimizations to do
//#define CPU_OPTS

#define REQ MPI_THREAD_FUNNELED

#endif //TIRAMISU_GEMV_PARAMS_H
