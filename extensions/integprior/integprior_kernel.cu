#include "../common/helper_cuda.h"
#include "../common/helper_math.h"

#include "integprior_kernel.h"
#include "kernel_dispatch.h"

void integprior_forward_cuda(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    const float* rayposim,
    const float* raydirim,
    float stepsize,
    const float* tminmaxim,
    const int* sortedobjid,
    const int* nodechildren,
    const float* nodeaabb,
    const float* tplate,
    const float* warp,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    float* rayrgbaim,
    float* priorim,
    float* raysatim,
    bool chlast,
    float fadescale,
    float fadeexp,
    int accum,
    float termthresh,
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

  auto fn = get_integprior_fwd_kernel_func(warp, accum, chlast);

  fn<<<gridsize, blocksize, 0, stream>>>(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      reinterpret_cast<const float3*>(rayposim),
      reinterpret_cast<const float3*>(raydirim),
      stepsize,
      reinterpret_cast<const float2*>(tminmaxim),
      reinterpret_cast<const int*>(sortedobjid),
      reinterpret_cast<const int2*>(nodechildren),
      reinterpret_cast<const float3*>(nodeaabb),
      reinterpret_cast<const float*>(tplate),
      reinterpret_cast<const float*>(warp),
      reinterpret_cast<const float3*>(primpos),
      reinterpret_cast<const float3*>(primrot),
      reinterpret_cast<const float3*>(primscale),
      reinterpret_cast<float*>(rayrgbaim),
      reinterpret_cast<float*>(priorim),
      reinterpret_cast<float3*>(raysatim),
      fadescale,
      fadeexp,
      termthresh);
}

void integprior_backward_cuda(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    const float* rayposim,
    const float* raydirim,
    float stepsize,
    const float* tminmaxim,
    const int* sortedobjid,
    const int* nodechildren,
    const float* nodeaabb,
    const float* tplate,
    const float* warp,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    const float* rayrgbaim,
    const float* raysatim,
    const float* grad_prior,
    float* grad_tplate,
    float* grad_warp,
    float* grad_primpos,
    float* grad_primrot,
    float* grad_primscale,
    bool chlast,
    float fadescale,
    float fadeexp,
    int accum,
    float termthresh,
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

  auto fn = get_integprior_bwd_kernel_func(warp, accum, chlast);

  fn<<<gridsize, blocksize, 0, stream>>>(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      reinterpret_cast<const float3*>(rayposim),
      reinterpret_cast<const float3*>(raydirim),
      stepsize,
      reinterpret_cast<const float2*>(tminmaxim),
      reinterpret_cast<const int*>(sortedobjid),
      reinterpret_cast<const int2*>(nodechildren),
      reinterpret_cast<const float3*>(nodeaabb),
      reinterpret_cast<const float*>(tplate),
      reinterpret_cast<const float*>(warp),
      reinterpret_cast<const float3*>(primpos),
      reinterpret_cast<const float3*>(primrot),
      reinterpret_cast<const float3*>(primscale),
      reinterpret_cast<const float*>(rayrgbaim),
      reinterpret_cast<const float3*>(raysatim),
      reinterpret_cast<const float*>(grad_prior),
      reinterpret_cast<float*>(grad_tplate),
      reinterpret_cast<float*>(grad_warp),
      reinterpret_cast<float3*>(grad_primpos),
      reinterpret_cast<float3*>(grad_primrot),
      reinterpret_cast<float3*>(grad_primscale),
      fadescale,
      fadeexp,
      termthresh);
}
