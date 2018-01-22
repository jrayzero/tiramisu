#define GPU
#define DIST
//#define CPU
#define CHECK

#ifndef CHECK
#define ROWS (int64_t)16000
#define COLS (int64_t)2000000
#define RESIDENT (int64_t)400
#define BLOCK_SIZE (int64_t)1000
#define ITERS 10
#else 
#define ROWS (int64_t)1600
#define COLS (int64_t)2000
#define RESIDENT (int64_t)40
#define BLOCK_SIZE (int64_t)100
#define ITERS 1
#endif
