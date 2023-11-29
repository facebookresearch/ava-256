#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../common/helper_cuda.h"
#include "../common/helper_math.h"
#include "../common/utils.h"

#include "primeval.h"
#include "utils.h"

#include "kernel_dispatch.h"
#include "mvpraymarch_kernel.h"

template <typename tplate_t>
void raymarch_forward_cuda(
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
    const tplate_t* tplate,
    const float* warp,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    tplate_t* rayrgbaim,
    float* raysatim,
    int* raytermim,
    float* t_im,
    bool chlast,
    float fadescale,
    float fadeexp,
    int accum,
    float termthresh,
    int griddim,
    int blocksizex,
    int blocksizey,
    bool broadcast_slab,
    cudaStream_t stream) {
  using tplate_vec4 = typename vec_t<tplate_t, 4>::type;

  dim3 blocksize(blocksizex, blocksizey);
  dim3 gridsize;
  if (griddim == 3) {
    gridsize = dim3((W + blocksize.x - 1) / blocksize.x, (H + blocksize.y - 1) / blocksize.y, N);
  } else if (griddim == 2) {
    gridsize = dim3((W + blocksize.x - 1) / blocksize.x, (N * H + blocksize.y - 1) / blocksize.y);
  } else if (griddim == 1) {
    gridsize = dim3((N * H * W + blocksize.x - 1) / blocksize.x);
  }

  auto fn = get_fwd_kernel_func<tplate_t>(warp, accum, chlast);

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
      reinterpret_cast<const tplate_t*>(tplate),
      reinterpret_cast<const float*>(warp),
      reinterpret_cast<const float3*>(primpos),
      reinterpret_cast<const float3*>(primrot),
      reinterpret_cast<const float3*>(primscale),
      reinterpret_cast<tplate_vec4*>(rayrgbaim),
      reinterpret_cast<float3*>(raysatim),
      reinterpret_cast<int2*>(raytermim),
      t_im,
      broadcast_slab,
      fadescale,
      fadeexp,
      termthresh);
}

void raymarch_backward_cuda(
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
    const int* raytermim,
    const float* grad_rayrgba,
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

  auto fn = get_bwd_kernel_func<float>(warp, accum, chlast);

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
      reinterpret_cast<const float4*>(rayrgbaim),
      reinterpret_cast<const float3*>(raysatim),
      reinterpret_cast<const int2*>(raytermim),
      reinterpret_cast<const float4*>(grad_rayrgba),
      reinterpret_cast<float*>(grad_tplate),
      reinterpret_cast<float*>(grad_warp),
      reinterpret_cast<float3*>(grad_primpos),
      reinterpret_cast<float3*>(grad_primrot),
      reinterpret_cast<float3*>(grad_primscale),
      fadescale,
      fadeexp,
      termthresh);
}

// Explicit instantiations for float + uint templates. This allows us to split
// the CUDA code from the PyTorch extension code and have the latter in its own
// cpp file.
void raymarch_forward_cuda_float(
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
    float* raysatim,
    int* raytermim,
    float* t_im,
    bool chlast,
    float fadescale,
    float fadeexp,
    int accum,
    float termthresh,
    int griddim,
    int blocksizex,
    int blocksizey,
    bool broadcast_slab,
    cudaStream_t stream) {
  raymarch_forward_cuda(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      rayposim,
      raydirim,
      stepsize,
      tminmaxim,
      sortedobjid,
      nodechildren,
      nodeaabb,
      tplate,
      warp,
      primpos,
      primrot,
      primscale,
      rayrgbaim,
      raysatim,
      raytermim,
      t_im,
      chlast,
      fadescale,
      fadeexp,
      accum,
      termthresh,
      griddim,
      blocksizex,
      blocksizey,
      broadcast_slab,
      stream);
}

void raymarch_forward_cuda_uint(
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
    const uint8_t* tplate,
    const float* warp,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    uint8_t* rayrgbaim,
    float* t_im,
    bool chlast,
    float fadescale,
    float fadeexp,
    int accum,
    float termthresh,
    int griddim,
    int blocksizex,
    int blocksizey,
    bool broadcast_slab,
    cudaStream_t stream) {
  raymarch_forward_cuda(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      rayposim,
      raydirim,
      stepsize,
      tminmaxim,
      sortedobjid,
      nodechildren,
      nodeaabb,
      tplate,
      warp,
      primpos,
      primrot,
      primscale,
      rayrgbaim,
      nullptr,
      nullptr,
      t_im,
      chlast,
      fadescale,
      fadeexp,
      accum,
      termthresh,
      griddim,
      blocksizex,
      blocksizey,
      broadcast_slab,
      stream);
}
