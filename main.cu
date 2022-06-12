#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define cudaCheck(ans)                     \
  {                                        \
    gpu_assert((ans), __FILE__, __LINE__); \
  }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#define BLOCK_SIZE 16

__global__ void kernel_matrix_mult(float *out, float *a, float *b, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float tmp_sum = 0;

  if (row < n && col < n)
  {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < n; i++)
    {
      tmp_sum += a[row * n + i] * b[i * n + col];
    }
    out[row * n + col] = tmp_sum;
  }
}

// Set up(and cleanup) for the matrix multiplication on the GPU
void gpu_matrix_mult(float *out, float *a, float *b, int N)
{
  int SIZE = N * N;
  float *d_a, *d_b, *d_out;

  // Allocate arrays in device memory
  cudaCheck(cudaMalloc((void **)&d_a, sizeof(float) * SIZE));
  cudaCheck(cudaMalloc((void **)&d_b, sizeof(float) * SIZE));
  cudaCheck(cudaMalloc((void **)&d_out, sizeof(float) * SIZE));

  // Copy data from the host memory to the device memory
  cudaCheck(cudaMemcpy(d_a, a, sizeof(float) * SIZE, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_b, b, sizeof(float) * SIZE, cudaMemcpyHostToDevice));

  dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_per_grid(N / BLOCK_SIZE, N / BLOCK_SIZE);

  printf("using %d threads per block\n", threads_per_block.x * threads_per_block.y);
  printf("using %d blocks per grid\n", blocks_per_grid.x * blocks_per_grid.y);

  kernel_matrix_mult<<<blocks_per_grid, threads_per_block>>>(d_out, d_a, d_b, N);
  cudaCheck(cudaPeekAtLastError());
  cudaCheck(cudaDeviceSynchronize());

  // Copy result from device memory to the host memory
  cudaCheck(cudaMemcpy(out, d_out, sizeof(float) * SIZE, cudaMemcpyDeviceToHost));
  cudaCheck(cudaDeviceSynchronize());

  // Free arrays in device memory
  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_out));
}

void inspect_gpu()
{
  int device;
  cudaGetDevice(&device);
  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  printf("---- GPU INFO -------\n");
  printf("\tusing %d multiprocessors\n", properties.multiProcessorCount);
  printf("\tmax blocks per processor: %d\n", properties.maxBlocksPerMultiProcessor);
  printf("\tmax threads per block: %d\n", properties.maxThreadsPerBlock);
  printf("\tmax threads per processor: %d\n\n", properties.maxThreadsPerMultiProcessor);
}

void cpu_matrix_mult(float *out, float *a, float *b, int N)
{
  for (int y = 0; y < N; y++)
  {
    for (int x = 0; x < N; x++)
    {
      float sum = 0.f;
      for (int n = 0; n < N; n++)
      {
        sum += a[y * N + n] * b[n * N + x];
      }
      out[y * N + x] = sum;
    }
  }
}

double mean_squared_error(float *a, float *b, int N)
{
  double err = 0;
  for (int y = 0; y < N; y++)
  {
    for (int x = 0; x < N; x++)
    {
      int i = y * N + x;
      err += pow(a[i] - b[i], 2);
    }
  }
  return err;
}

int main()
{
  int N = 1024;
  int SIZE = N * N;

  float *a = (float *)malloc(sizeof(float) * SIZE);
  float *b = (float *)malloc(sizeof(float) * SIZE);

  // Initialize matrices on the host
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      a[i * N + j] = sin(i);
      b[i * N + j] = cos(j);
    }
  }

  inspect_gpu();

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

  float *out = (float *)malloc(sizeof(float) * SIZE);
  cudaCheck(cudaEventRecord(start));
  cudaCheck(cudaEventSynchronize(start));
  gpu_matrix_mult(out, a, b, N);
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  float gpu_msec_total = 0.0f;
  cudaCheck(cudaEventElapsedTime(&gpu_msec_total, start, stop));

  float *cpu_out = (float *)malloc(sizeof(float) * SIZE);
  cudaCheck(cudaEventRecord(start));
  cudaCheck(cudaEventSynchronize(start));
  cpu_matrix_mult(cpu_out, a, b, N);
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  float cpu_msec_total = 0.0f;
  cudaCheck(cudaEventElapsedTime(&cpu_msec_total, start, stop));

  double err = mean_squared_error(out, cpu_out, N);

  printf("Mean squared error: %f\n", err);
  printf("Time elapsed GPU: %.2fms. CPU: %.2fms\n", gpu_msec_total, cpu_msec_total);

  // Deallocate host memory
  free(a);
  free(b);
  free(out);
  free(cpu_out);
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));
}
