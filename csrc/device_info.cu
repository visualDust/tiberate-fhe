#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int maxThreadsPerBlock;
  cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

  int maxBlocksPerSM;
  cudaDeviceGetAttribute(
      &maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, 0);

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  printf("Maximum number of threads per block: %d\n", maxThreadsPerBlock);
  printf("Maximum number of blocks per SM: %d\n", maxBlocksPerSM);
  printf("Number of SMs: %d\n", numSMs);

  int maxGridSize[3];
  cudaDeviceGetAttribute(&maxGridSize[0], cudaDevAttrMaxGridDimX, 0);
  cudaDeviceGetAttribute(&maxGridSize[1], cudaDevAttrMaxGridDimY, 0);
  cudaDeviceGetAttribute(&maxGridSize[2], cudaDevAttrMaxGridDimZ, 0);
  printf("Maximum grid size: %d x %d x %d\n",
         maxGridSize[0],
         maxGridSize[1],
         maxGridSize[2]);

  int maxSharedMemory;
  cudaDeviceGetAttribute(
      &maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Maximum shared memory per block: %d bytes\n", maxSharedMemory);

  int warpSize;
  cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
  printf("Warp size: %d\n", warpSize);

  int maxRegistersPerBlock;
  cudaDeviceGetAttribute(
      &maxRegistersPerBlock, cudaDevAttrMaxRegistersPerBlock, 0);
  printf("Maximum number of registers per block: %d\n", maxRegistersPerBlock);

  int maxSharedMemoryPerMultiprocessor;
  cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                         0);
  printf("Maximum shared memory per multiprocessor: %d bytes\n",
         maxSharedMemoryPerMultiprocessor);

  int maxRegistersPerMultiprocessor;
  cudaDeviceGetAttribute(&maxRegistersPerMultiprocessor,
                         cudaDevAttrMaxRegistersPerMultiprocessor,
                         0);
  printf("Maximum number of registers per multiprocessor: %d\n",
         maxRegistersPerMultiprocessor);

  int maxThreadsPerMultiprocessor;
  cudaDeviceGetAttribute(
      &maxThreadsPerMultiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  printf("Maximum number of threads per multiprocessor: %d\n",
         maxThreadsPerMultiprocessor);

  int memoryClockRate;
  cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
  printf("Memory clock rate: %d kHz\n", memoryClockRate);

  int memoryBusWidth;
  cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
  printf("Memory bus width: %d bits\n", memoryBusWidth);

  int L2CacheSize;
  cudaDeviceGetAttribute(&L2CacheSize, cudaDevAttrL2CacheSize, 0);
  printf("L2 cache size: %d bytes\n", L2CacheSize);

  int Major;
  cudaDeviceGetAttribute(&Major, cudaDevAttrComputeCapabilityMajor, 0);
  int Minor;
  cudaDeviceGetAttribute(&Minor, cudaDevAttrComputeCapabilityMinor, 0);
  printf("Compute capability: %d.%d\n", Major, Minor);
}
