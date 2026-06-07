#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 16
#define SKEW 8
#define LDA_S (BM + SKEW)
#define LDB_K (BK + SKEW)
#define THREADS 256
#define WARP_M 4
#define WARP_N 2
#define WM_TILES (BM / (WARP_M * 16))
#define WN_TILES (BN / (WARP_N * 16))

#define A_VEC_PER_THREAD ((BM * BK) / (4 * THREADS))
#define B_VEC_PER_THREAD ((BK * BN) / (4 * THREADS))

__global__ void __launch_bounds__(THREADS) gemm_tc(int m, int n, int k,
                                                   const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C)
{
  const int m0 = BM * blockIdx.x;
  const int n0 = BN * blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int warpM = warp_id / WARP_N;
  const int warpN = warp_id % WARP_N;

  // As[buf][k][m]: m contiguous, col_major fragments
  // Bs[buf][n][k]: k contiguous, col_major fragments (transposed to fix store conflicts)
  __shared__ half As[2][BK][LDA_S];
  __shared__ half Bs[2][BN][LDB_K];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM_TILES][WN_TILES];
#pragma unroll
  for (int r = 0; r < WM_TILES; r++)
#pragma unroll
    for (int c = 0; c < WN_TILES; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  const int num_kt = k / BK;

  int a_ksub[A_VEC_PER_THREAD], a_m[A_VEC_PER_THREAD];
#pragma unroll
  for (int i = 0; i < A_VEC_PER_THREAD; i++)
  {
    int idx = tid + i * THREADS;
    a_ksub[i] = idx / (BM / 4);
    a_m[i] = (idx % (BM / 4)) * 4;
  }

  int b_n[B_VEC_PER_THREAD], b_k[B_VEC_PER_THREAD];
#pragma unroll
  for (int i = 0; i < B_VEC_PER_THREAD; i++)
  {
    int idx = tid + i * THREADS;
    b_n[i] = idx / (BK / 4);
    b_k[i] = (idx % (BK / 4)) * 4;
  }

  float4 rA[A_VEC_PER_THREAD];
  float4 rB[B_VEC_PER_THREAD];

  auto load_regs = [&](int koff)
  {
#pragma unroll
    for (int i = 0; i < A_VEC_PER_THREAD; i++)
    {
      size_t off = (size_t)(m0 + a_m[i]) + (size_t)(koff + a_ksub[i]) * m;
      rA[i] = *reinterpret_cast<const float4*>(A + off);
    }
#pragma unroll
    for (int i = 0; i < B_VEC_PER_THREAD; i++)
    {
      size_t off = (size_t)(koff + b_k[i]) + (size_t)(n0 + b_n[i]) * k;
      rB[i] = *reinterpret_cast<const float4*>(B + off);
    }
  };

  auto store_regs = [&](int buf)
  {
#pragma unroll
    for (int i = 0; i < A_VEC_PER_THREAD; i++)
    {
      half2* d = reinterpret_cast<half2*>(&As[buf][a_ksub[i]][a_m[i]]);
      d[0] = __floats2half2_rn(rA[i].x, rA[i].y);
      d[1] = __floats2half2_rn(rA[i].z, rA[i].w);
    }
#pragma unroll
    for (int i = 0; i < B_VEC_PER_THREAD; i++)
    {
      half2* d = reinterpret_cast<half2*>(&Bs[buf][b_n[i]][b_k[i]]);
      d[0] = __floats2half2_rn(rB[i].x, rB[i].y);
      d[1] = __floats2half2_rn(rB[i].z, rB[i].w);
    }
  };

  load_regs(0);
  store_regs(0);
  __syncthreads();

  for (int kt = 0; kt < num_kt; kt++)
  {
    const int buf = kt & 1;

    if (kt + 1 < num_kt)
      load_regs((kt + 1) * BK);

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16)
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[WM_TILES];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[WN_TILES];
#pragma unroll
      for (int r = 0; r < WM_TILES; r++)
      {
        const half* aptr = &As[buf][kk][warpM * (WM_TILES * 16) + r * 16];
        wmma::load_matrix_sync(a_frag[r], aptr, LDA_S);
      }
#pragma unroll
      for (int c = 0; c < WN_TILES; c++)
      {
        const half* bptr = &Bs[buf][warpN * (WN_TILES * 16) + c * 16][kk];
        wmma::load_matrix_sync(b_frag[c], bptr, LDB_K);
      }
#pragma unroll
      for (int r = 0; r < WM_TILES; r++)
#pragma unroll
        for (int c = 0; c < WN_TILES; c++)
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
    }

    if (kt + 1 < num_kt)
      store_regs((kt + 1) & 1);

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < WM_TILES; r++)
  {
#pragma unroll
    for (int c = 0; c < WN_TILES; c++)
    {
      int c_m = m0 + warpM * (WM_TILES * 16) + r * 16;
      int c_n = n0 + warpN * (WN_TILES * 16) + c * 16;
      wmma::store_matrix_sync(&C[(size_t)c_m + (size_t)c_n * m],
                              acc[r][c], m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char** argv)
{
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, (size_t)m * k * sizeof(float));
  cudaMallocManaged(&B, (size_t)k * n * sizeof(float));
  cudaMallocManaged(&C, (size_t)m * n * sizeof(float));
  cudaMallocManaged(&C2, (size_t)m * n * sizeof(float));
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      A[(size_t)k * i + j] = drand48();
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      B[(size_t)n * i + j] = drand48();
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      C[(size_t)m * i + j] = C2[(size_t)m * i + j] = 0;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt + 2; i++)
  {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k, &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block(THREADS);
  dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN);
  for (int i = 0; i < Nt + 2; i++)
  {
    if (i == 2) tic = chrono::steady_clock::now();
    gemm_tc<<<grid, block>>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tmine = chrono::duration<double>(toc - tic).count() / Nt;
  double mine_flops = double(num_flops) / tmine / 1.0e9;

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(e));

  printf("CUBLAS: %.2f Gflops, MINE: %.2f Gflops  (%.1f%% of cuBLAS)\n",
         cublas_flops, mine_flops, 100.0 * mine_flops / cublas_flops);

  double err = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      err += fabs(C[(size_t)m * i + j] - C2[(size_t)m * i + j]);
  printf("error: %lf\n", err / n / m);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}