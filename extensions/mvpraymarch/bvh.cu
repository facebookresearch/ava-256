#include <cmath>
#include <cstdio>

#include "../common/helper_math.h"

//#include <cub/device/device_radix_sort.cuh>

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expand_bits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z) {
  x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
  y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
  z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
  unsigned int xx = expand_bits((unsigned int)x);
  unsigned int yy = expand_bits((unsigned int)y);
  unsigned int zz = expand_bits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

__global__ void compute_morton_kernel(int N, int K, float4* center, int* code) {
  const int count = N * K;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    const int k = index % K;
    const int n = index / K;

    float4 c = center[n * K + k];
    code[n * K + k] = morton3D(c.x, c.y, c.z);
  }
}

__forceinline__ __device__ int delta(int* sortedcodes, int x, int y, int K) {
  if (x >= 0 && x <= K - 1 && y >= 0 && y <= K - 1) {
    return sortedcodes[x] == sortedcodes[y] ? 32 + __clz(x ^ y)
                                            : __clz(sortedcodes[x] ^ sortedcodes[y]);
  }
  return -1;
}

__forceinline__ __device__ int sign(int x) {
  return (int)(x > 0) - (int)(x < 0);
}

__device__ int find_split(int* sortedcodes, int first, int last, int K) {
  float commonPrefix = delta(sortedcodes, first, last, K);
  int split = first;
  int step = last - first;

  do {
    step = (step + 1) >> 1; // exponential decrease
    int newSplit = split + step; // proposed new position

    if (newSplit < last) {
      int splitPrefix = delta(sortedcodes, first, newSplit, K);
      if (splitPrefix > commonPrefix) {
        split = newSplit; // accept proposal
      }
    }
  } while (step > 1);

  return split;
}

__device__ int2 determine_range(int* sortedcodes, int K, int idx) {
  int d = sign(delta(sortedcodes, idx, idx + 1, K) - delta(sortedcodes, idx, idx - 1, K));
  int dmin = delta(sortedcodes, idx, idx - d, K);
  int lmax = 2;
  while (delta(sortedcodes, idx, idx + lmax * d, K) > dmin) {
    lmax = lmax * 2;
  }
  int l = 0;
  for (int t = lmax / 2; t >= 1; t /= 2) {
    if (delta(sortedcodes, idx, idx + (l + t) * d, K) > dmin) {
      l += t;
    }
  }
  int j = idx + l * d;
  int2 range;
  range.x = min(idx, j);
  range.y = max(idx, j);
  return range;
}

__global__ void
build_tree_kernel(int N, int K, int* sortedcodes, int2* nodechildren, int* nodeparent) {
  const int count = N * (K + K - 1);
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    const int k = index % (K + K - 1);
    const int n = index / (K + K - 1);

    if (k >= K - 1) {
      // leaf
      nodechildren[n * (K + K - 1) + k] = make_int2(-(k - (K - 1)) - 1, -(k - (K - 1)) - 2);
    } else {
      // internal node

      // find out which range of objects the node corresponds to
      int2 range = determine_range(sortedcodes + n * K, K, k);
      int first = range.x;
      int last = range.y;

      // determine where to split the range
      int split = find_split(sortedcodes + n * K, first, last, K);

      // select childA
      int childa = split == first ? (K - 1) + split : split;

      // select childB
      int childb = split + 1 == last ? (K - 1) + split + 1 : split + 1;

      // record parent-child relationships
      nodechildren[n * (K + K - 1) + k] = make_int2(childa, childb);
      nodeparent[n * (K + K - 1) + childa] = k;
      nodeparent[n * (K + K - 1) + childb] = k;
    }
  }
}

__global__ void compute_aabb2_kernel(
    int N,
    int K,
    int* sortedobjid,
    float3* primpos,
    float3* primrot,
    float3* primscale,
    int2* nodechildren,
    int* nodeparent,
    float3* nodeaabb,
    int* atom) {
  const int count = N * K;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += blockDim.x * gridDim.x) {
    const int k = index % K;
    const int n = index / K;

    int kk = sortedobjid[n * K + k];

    float3 w1t = primpos[n * K + kk];
    float3 w1r0 = primrot[(n * K + kk) * 3 + 0];
    float3 w1r1 = primrot[(n * K + kk) * 3 + 1];
    float3 w1r2 = primrot[(n * K + kk) * 3 + 2];
    float3 w1s = primscale[n * K + kk];

    float3 p;
    p = make_float3(-1.f, -1.f, -1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    float3 pmin = p;
    float3 pmax = p;

    p = make_float3(1.f, -1.f, -1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(-1.f, 1.f, -1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(1.f, 1.f, -1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(-1.f, -1.f, 1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(1.f, -1.f, 1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(-1.f, 1.f, 1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(1.f, 1.f, 1.f) / w1s;
    p = make_float3(dot(p, w1r0), dot(p, w1r1), dot(p, w1r2)) + w1t;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    nodeaabb[n * (K + K - 1) * 2 + ((K - 1) + k) * 2 + 0] = pmin;
    nodeaabb[n * (K + K - 1) * 2 + ((K - 1) + k) * 2 + 1] = pmax;

    int node = nodeparent[n * (K + K - 1) + ((K - 1) + k)];

    while (node != -1 && atomicCAS(&atom[n * (K - 1) + node], 0, 1) == 1) {
      int2 children = nodechildren[n * (K + K - 1) + node];
      float3 laabbmin = nodeaabb[n * (K + K - 1) * 2 + children.x * 2 + 0];
      float3 laabbmax = nodeaabb[n * (K + K - 1) * 2 + children.x * 2 + 1];
      float3 raabbmin = nodeaabb[n * (K + K - 1) * 2 + children.y * 2 + 0];
      float3 raabbmax = nodeaabb[n * (K + K - 1) * 2 + children.y * 2 + 1];

      float3 aabbmin = fminf(laabbmin, raabbmin);
      float3 aabbmax = fmaxf(laabbmax, raabbmax);

      nodeaabb[n * (K + K - 1) * 2 + node * 2 + 0] = aabbmin;
      nodeaabb[n * (K + K - 1) * 2 + node * 2 + 1] = aabbmax;

      node = nodeparent[n * (K + K - 1) + node];

      __threadfence();
    }
  }
}

void compute_morton_cuda(int N, int K, float* center, int* code, cudaStream_t stream) {
  int count = N * K;
  int nthreads = 512;
  int nblocks = (count + nthreads - 1) / nthreads;
  compute_morton_kernel<<<nblocks, nthreads, 0, stream>>>(
      N, K, reinterpret_cast<float4*>(center), code);
}

void build_tree_cuda(
    int N,
    int K,
    int* sortedcode,
    int* nodechildren,
    int* nodeparent,
    cudaStream_t stream) {
  int count = N * (K + K - 1);
  int nthreads = 512;
  int nblocks = (count + nthreads - 1) / nthreads;
  build_tree_kernel<<<nblocks, nthreads, 0, stream>>>(
      N, K, sortedcode, reinterpret_cast<int2*>(nodechildren), nodeparent);
}

void compute_aabb2_cuda(
    int N,
    int K,
    int* sortedobjid,
    float* primpos,
    float* primrot,
    float* primscale,
    int* nodechildren,
    int* nodeparent,
    float* nodeaabb,
    int* atom,
    cudaStream_t stream) {
  int count = N * K;
  int nthreads = 512;
  int nblocks = (count + nthreads - 1) / nthreads;
  compute_aabb2_kernel<<<nblocks, nthreads, 0, stream>>>(
      N,
      K,
      sortedobjid,
      reinterpret_cast<float3*>(primpos),
      reinterpret_cast<float3*>(primrot),
      reinterpret_cast<float3*>(primscale),
      reinterpret_cast<int2*>(nodechildren),
      nodeparent,
      reinterpret_cast<float3*>(nodeaabb),
      atom);
}
