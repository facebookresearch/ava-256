#include "../common/helper_cuda.h"
#include "../common/helper_math.h"

#include "utils.h"

#include "primintersection_kernel.h"

void primintersection_forward_cuda(
    int N,
    int H,
    int W,
    int K,
    const float* rayposim,
    const float* raydirim,
    const int* sortedobjid,
    const int* nodechildren,
    const float* nodeaabb,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    float* raystepsim,
    int griddim,
    int blocksizex,
    int blocksizey,
    cudaStream_t stream) {
  dim3 blocksize(blocksizex, blocksizey);
  dim3 gridsize;
  if (griddim == 3) {
    gridsize = dim3((W + blocksize.x - 1) / blocksize.x, (H + blocksize.y - 1) / blocksize.y, N);
  } else if (griddim == 2) {
    gridsize = dim3((W + blocksize.x - 1) / blocksize.x, (N * H + blocksize.y - 1) / blocksize.y);
  } else if (griddim == 1) {
    gridsize = dim3((N * H * W + blocksize.x - 1) / blocksize.x);
  }

  primintersection_forward_kernel<<<gridsize, blocksize, 0, stream>>>(
      N,
      H,
      W,
      K,
      reinterpret_cast<const float3*>(rayposim),
      reinterpret_cast<const float3*>(raydirim),
      reinterpret_cast<const int*>(sortedobjid),
      reinterpret_cast<const int2*>(nodechildren),
      reinterpret_cast<const float3*>(nodeaabb),
      reinterpret_cast<const float3*>(primpos),
      reinterpret_cast<const float3*>(primrot),
      reinterpret_cast<const float3*>(primscale),
      reinterpret_cast<float*>(raystepsim));
}

void primintersection_backward_cuda(
    int N,
    int H,
    int W,
    int K,
    const float* rayposim,
    const float* raydirim,
    const int* sortedobjid,
    const int* nodechildren,
    const float* nodeaabb,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    const float* raystepsim,
    const float* grad_raysteps,
    float* grad_primpos,
    float* grad_primrot,
    float* grad_primscale,
    int griddim,
    int blocksizex,
    int blocksizey,
    cudaStream_t stream) {
  dim3 blocksize(blocksizex, blocksizey);
  dim3 gridsize;
  if (griddim == 3) {
    gridsize = dim3((W + blocksize.x - 1) / blocksize.x, (H + blocksize.y - 1) / blocksize.y, N);
  } else if (griddim == 2) {
    gridsize = dim3((W + blocksize.x - 1) / blocksize.x, (N * H + blocksize.y - 1) / blocksize.y);
  } else if (griddim == 1) {
    gridsize = dim3((N * H * W + blocksize.x - 1) / blocksize.x);
  }

  primintersection_backward_kernel<<<gridsize, blocksize, 0, stream>>>(
      N,
      H,
      W,
      K,
      reinterpret_cast<const float3*>(rayposim),
      reinterpret_cast<const float3*>(raydirim),
      reinterpret_cast<const int*>(sortedobjid),
      reinterpret_cast<const int2*>(nodechildren),
      reinterpret_cast<const float3*>(nodeaabb),
      reinterpret_cast<const float3*>(primpos),
      reinterpret_cast<const float3*>(primrot),
      reinterpret_cast<const float3*>(primscale),
      reinterpret_cast<const float*>(raystepsim),
      reinterpret_cast<const float*>(grad_raysteps),
      reinterpret_cast<float3*>(grad_primpos),
      reinterpret_cast<float3*>(grad_primrot),
      reinterpret_cast<float3*>(grad_primscale));
}
