#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H

#define MAX_CONST_BYTES (50 * 1024)
extern __device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];
#endif  // SHARED_CONSTANTS_H
