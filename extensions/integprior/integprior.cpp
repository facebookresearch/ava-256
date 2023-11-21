#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../common/utils.h"

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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

  c10::cuda::CUDAGuard deviceGuard{rayposim.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

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
      stream);

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

  c10::cuda::CUDAGuard deviceGuard{rayposim.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

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
      stream);

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
