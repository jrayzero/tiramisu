//#define GPU
#define COOP
//#define DIST
//#define CPU
//#define CHECK

//#define GPU_PROCS 1
//#define CPU_PROCS 1

#define ROWS (int64_t)16000
#define GPU_ROWS (int64_t)12800
#define CPU_ROWS (int64_t)3200
#define CPU_PROCS (int64_t)1
#define GPU_PROCS (int64_t)1

#define PROCS (int64_t)2

#ifndef CHECK
#define COLS (int64_t)2000000//00
#define RESIDENT (int64_t)400
#define BLOCK_SIZE (int64_t)1000
#define ITERS 1
#else 
#define ROWS (int64_t)16000
#define COLS (int64_t)20000
#define RESIDENT (int64_t)40
#define BLOCK_SIZE (int64_t)100
#define ITERS 10
#endif
