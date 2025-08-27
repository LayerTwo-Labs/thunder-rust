#ifndef DEBUG_CUH
#define DEBUG_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
CUDA error checking macro with immediate failure detection
Usage: CUDA(cudaMalloc(&ptr, size));
*/
#define CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        fflush(stderr); \
        abort(); \
    } \
} while(0)

/*
Assertion macro with file/line information
Usage: ASSERT(ptr != nullptr, "Pointer must not be null");
*/
#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "ASSERT FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
        fflush(stderr); \
        abort(); \
    } \
} while(0)

/*
Check for CUDA errors after kernel launches
Usage: After any kernel launch, call CHECK_CUDA_KERNEL();
*/
#define CHECK_CUDA_KERNEL() do { \
    CUDA(cudaGetLastError()); \
    CUDA(cudaDeviceSynchronize()); \
} while(0)

#endif // DEBUG_CUH