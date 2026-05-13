#include <cstdio>
#include <cstdlib>
#include <vector>
#include<cuda_runtime.h>
#include<thrust/device_ptr.h>
#include<thrust/scan.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_key, *d_bucket, *d_prefix;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));
  cudaMalloc(&d_prefix, range * sizeof(int));
  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);

  int blockSize = 256;

  initBucket<<<(range + blockSize - 1) / blockSize, blockSize>>>(d_bucket, range);
  countBuckets<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_key, d_bucket, n);

  thrust::device_ptr<int> dp_bucket(d_bucket);
  thrust::device_ptr<int> dp_prefix(d_prefix);
  thrust::exclusive_scan(dp_bucket, dp_bucket + range, dp_prefix);

  writeback<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_key, d_prefix, n, range);

  cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(d_key);
  cudaFree(d_bucket);
  cudaFree(d_prefix);
  return 0;
}
