//
// Created by Jessica Ray on 1/17/18.
//

#ifndef TIRAMISU_GEMV_PARAMS_H
#define TIRAMISU_GEMV_PARAMS_H

//#define CPU
#define GPU

#define ROWS (int64_t)10000
#define COLS (int64_t)2000
#define ITERS 1
#define CHECK_RESULTS

// types of optimizations to do
#define CPU_OPTS

#define REQ MPI_THREAD_FUNNELED

#endif //TIRAMISU_GEMV_PARAMS_H
