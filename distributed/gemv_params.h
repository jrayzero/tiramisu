//
// Created by Jessica Ray on 1/17/18.
//

#ifndef TIRAMISU_GEMV_PARAMS_H
#define TIRAMISU_GEMV_PARAMS_H

//#define CPU
#define GPU

#define CHECK_RESULTS

#ifndef CHECK_RESULTS
#define ROWS (int64_t)1000000
#define COLS (int64_t)80000
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
