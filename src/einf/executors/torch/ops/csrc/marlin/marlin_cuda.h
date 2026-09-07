#include <cuda_runtime.h>

// Prototype for the vendored Marlin kernel (marlin_cuda_kernel.cu).
// A: fp16 [M, K] row-major; B: int32 [K/16, N*2] Marlin qweight;
// C: fp16 [M, N]; s: fp16 [K/groupsize, N] logical scales;
// workspace: int32 of at least N/128*max_par entries.
int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);
