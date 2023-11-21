#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../common/helper_cuda.h"
#include "../common/helper_math.h"

#include "utils.h"

#include "integprior_kernel.h"
#include "kernel_dispatch.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA((x));     \
  CHECK_CONTIGUOUS((x))

template <typename tplate_t>
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
    const tplate_t* tplate,
    const float* warp,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    tplate_t* rayrgbaim,
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

  auto fn = get_integprior_fwd_kernel_func<tplate_t>(warp, accum, chlast);

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
      reinterpret_cast<tplate_t*>(rayrgbaim),
      reinterpret_cast<float*>(priorim),
      reinterpret_cast<float3*>(raysatim),
      fadescale,
      fadeexp,
      termthresh);
}

template <typename tplate_t>
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
    const tplate_t* tplate,
    const float* warp,
    const float* primpos,
    const float* primrot,
    const float* primscale,
    const tplate_t* rayrgbaim,
    const float* raysatim,
    const float* grad_prior,
    tplate_t* grad_tplate,
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

  auto fn = get_integprior_bwd_kernel_func<tplate_t>(warp, accum, chlast);

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
      reinterpret_cast<const tplate_t*>(rayrgbaim),
      reinterpret_cast<const float3*>(raysatim),
      reinterpret_cast<const float*>(grad_prior),
      reinterpret_cast<tplate_t*>(grad_tplate),
      reinterpret_cast<float*>(grad_warp),
      reinterpret_cast<float3*>(grad_primpos),
      reinterpret_cast<float3*>(grad_primrot),
      reinterpret_cast<float3*>(grad_primscale),
      fadescale,
      fadeexp,
      termthresh);
}

std::vector<torch::Tensor> integprior_forward(
    const torch::Tensor rayposim,
    const torch::Tensor raydirim,
    double stepsize,
    const torch::Tensor tminmaxim,
    const torch::optional<torch::Tensor> sortedobjid,
    const torch::optional<torch::Tensor> nodechildren,
    const torch::optional<torch::Tensor> nodeaabb,
    const torch::Tensor tplate,
    const torch::optional<torch::Tensor> warp,
    const torch::Tensor primpos,
    const torch::Tensor primrot,
    const torch::Tensor primscale,
    const torch::Tensor rayrgbaim,
    torch::Tensor priorim,
    torch::Tensor raysatim,
    bool chlast = false,
    double fadescale = 8.f,
    double fadeexp = 8.f,
    int64_t accum = 0,
    double termthresh = 0.f,
    int64_t griddim = 3,
    int64_t blocksizex = 256,
    int64_t blocksizey = 1) {
  CHECK_INPUT(rayposim);
  CHECK_INPUT(raydirim);
  CHECK_INPUT(tminmaxim);
  if (sortedobjid) {
    CHECK_INPUT(*sortedobjid);
  }
  if (nodechildren) {
    CHECK_INPUT(*nodechildren);
  }
  if (nodeaabb) {
    CHECK_INPUT(*nodeaabb);
  }
  CHECK_INPUT(tplate);
  if (warp) {
    CHECK_INPUT(*warp);
  }
  CHECK_INPUT(primpos);
  CHECK_INPUT(primrot);
  CHECK_INPUT(primscale);
  CHECK_INPUT(rayrgbaim);
  CHECK_INPUT(priorim);
  CHECK_INPUT(raysatim);

  const int N = rayposim.size(0);
  const int H = rayposim.size(1);
  const int W = rayposim.size(2);
  const int K = primpos.size(1);
  const int MD = tplate.size(chlast ? 2 : 3);
  const int MH = tplate.size(chlast ? 3 : 4);
  const int MW = tplate.size(chlast ? 4 : 5);

  integprior_forward_cuda(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      reinterpret_cast<const float*>(rayposim.data_ptr()),
      reinterpret_cast<const float*>(raydirim.data_ptr()),
      stepsize,
      reinterpret_cast<const float*>(tminmaxim.data_ptr()),
      sortedobjid ? reinterpret_cast<const int*>(sortedobjid->data_ptr()) : nullptr,
      nodechildren ? reinterpret_cast<const int*>(nodechildren->data_ptr()) : nullptr,
      nodeaabb ? reinterpret_cast<const float*>(nodeaabb->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(tplate.data_ptr()),
      warp ? reinterpret_cast<const float*>(warp->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(primpos.data_ptr()),
      reinterpret_cast<const float*>(primrot.data_ptr()),
      reinterpret_cast<const float*>(primscale.data_ptr()),
      reinterpret_cast<float*>(rayrgbaim.data_ptr()),
      reinterpret_cast<float*>(priorim.data_ptr()),
      reinterpret_cast<float*>(raysatim.data_ptr()),
      chlast,
      fadescale,
      fadeexp,
      accum,
      termthresh,
      griddim,
      blocksizex,
      blocksizey,
      0);

  return {};
}

std::vector<torch::Tensor> integprior_backward(
    const torch::Tensor rayposim,
    const torch::Tensor raydirim,
    double stepsize,
    const torch::Tensor tminmaxim,
    const torch::optional<torch::Tensor> sortedobjid,
    const torch::optional<torch::Tensor> nodechildren,
    const torch::optional<torch::Tensor> nodeaabb,
    const torch::Tensor tplate,
    const torch::optional<torch::Tensor> warp,
    const torch::Tensor primpos,
    const torch::Tensor primrot,
    const torch::Tensor primscale,
    const torch::Tensor rayrgbaim,
    const torch::Tensor raysatim,
    const torch::Tensor grad_prior,
    torch::Tensor grad_tplate,
    torch::optional<torch::Tensor> grad_warp,
    torch::Tensor grad_primpos,
    torch::Tensor grad_primrot,
    torch::Tensor grad_primscale,
    bool chlast = false,
    double fadescale = 8.f,
    double fadeexp = 8.f,
    int64_t accum = 0,
    double termthresh = 0.f,
    int64_t griddim = 3,
    int64_t blocksizex = 256,
    int64_t blocksizey = 1) {
  CHECK_INPUT(rayposim);
  CHECK_INPUT(raydirim);
  CHECK_INPUT(tminmaxim);
  if (sortedobjid) {
    CHECK_INPUT(*sortedobjid);
  }
  if (nodechildren) {
    CHECK_INPUT(*nodechildren);
  }
  if (nodeaabb) {
    CHECK_INPUT(*nodeaabb);
  }
  CHECK_INPUT(tplate);
  if (warp) {
    CHECK_INPUT(*warp);
  }
  CHECK_INPUT(primpos);
  CHECK_INPUT(primrot);
  CHECK_INPUT(primscale);
  CHECK_INPUT(rayrgbaim);
  CHECK_INPUT(raysatim);
  CHECK_INPUT(grad_prior);
  CHECK_INPUT(grad_tplate);
  if (grad_warp) {
    CHECK_INPUT(*grad_warp);
  }
  CHECK_INPUT(grad_primpos);
  CHECK_INPUT(grad_primrot);
  CHECK_INPUT(grad_primscale);

  const int N = rayposim.size(0);
  const int H = rayposim.size(1);
  const int W = rayposim.size(2);
  const int K = primpos.size(1);
  const int MD = tplate.size(chlast ? 2 : 3);
  const int MH = tplate.size(chlast ? 3 : 4);
  const int MW = tplate.size(chlast ? 4 : 5);

  integprior_backward_cuda(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      reinterpret_cast<const float*>(rayposim.data_ptr()),
      reinterpret_cast<const float*>(raydirim.data_ptr()),
      stepsize,
      reinterpret_cast<const float*>(tminmaxim.data_ptr()),
      sortedobjid ? reinterpret_cast<const int*>(sortedobjid->data_ptr()) : nullptr,
      nodechildren ? reinterpret_cast<const int*>(nodechildren->data_ptr()) : nullptr,
      nodeaabb ? reinterpret_cast<const float*>(nodeaabb->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(tplate.data_ptr()),
      warp ? reinterpret_cast<const float*>(warp->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(primpos.data_ptr()),
      reinterpret_cast<const float*>(primrot.data_ptr()),
      reinterpret_cast<const float*>(primscale.data_ptr()),
      reinterpret_cast<const float*>(rayrgbaim.data_ptr()),
      reinterpret_cast<const float*>(raysatim.data_ptr()),
      reinterpret_cast<const float*>(grad_prior.data_ptr()),
      reinterpret_cast<float*>(grad_tplate.data_ptr()),
      grad_warp ? reinterpret_cast<float*>(grad_warp->data_ptr()) : nullptr,
      reinterpret_cast<float*>(grad_primpos.data_ptr()),
      reinterpret_cast<float*>(grad_primrot.data_ptr()),
      reinterpret_cast<float*>(grad_primscale.data_ptr()),
      chlast,
      fadescale,
      fadeexp,
      accum,
      termthresh,
      griddim,
      blocksizex,
      blocksizey,
      0);

  return {};
}

#ifndef NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integprior_forward", &integprior_forward, "integprior forward (CUDA)");
  m.def("integprior_backward", &integprior_backward, "integprior backward (CUDA)");
}
#endif

TORCH_LIBRARY(integprior_ext, m) {
  m.def("integprior_forward", &integprior_forward);
  m.def("integprior_backward", &integprior_backward);
}
