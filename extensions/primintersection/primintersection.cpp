#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../common/utils.h"

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
    cudaStream_t stream);

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
    cudaStream_t stream);

std::vector<torch::Tensor> primintersection_forward(
    const torch::Tensor rayposim,
    const torch::Tensor raydirim,
    const torch::optional<torch::Tensor> sortedobjid,
    const torch::optional<torch::Tensor> nodechildren,
    const torch::optional<torch::Tensor> nodeaabb,
    const torch::Tensor primpos,
    const torch::Tensor primrot,
    const torch::Tensor primscale,
    torch::Tensor raystepsim,
    int64_t griddim = 3,
    int64_t blocksizex = 256,
    int64_t blocksizey = 1) {
  CHECK_INPUT(rayposim);
  CHECK_INPUT(raydirim);
  if (sortedobjid) {
    CHECK_INPUT(*sortedobjid);
  }
  if (nodechildren) {
    CHECK_INPUT(*nodechildren);
  }
  if (nodeaabb) {
    CHECK_INPUT(*nodeaabb);
  }
  CHECK_INPUT(primpos);
  CHECK_INPUT(primrot);
  CHECK_INPUT(primscale);
  CHECK_INPUT(raystepsim);

  const int N = rayposim.size(0);
  const int H = rayposim.size(1);
  const int W = rayposim.size(2);
  const int K = primpos.size(1);

  c10::cuda::CUDAGuard deviceGuard{rayposim.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  primintersection_forward_cuda(
      N,
      H,
      W,
      K,
      reinterpret_cast<const float*>(rayposim.data_ptr()),
      reinterpret_cast<const float*>(raydirim.data_ptr()),
      sortedobjid ? reinterpret_cast<const int*>(sortedobjid->data_ptr()) : nullptr,
      nodechildren ? reinterpret_cast<const int*>(nodechildren->data_ptr()) : nullptr,
      nodeaabb ? reinterpret_cast<const float*>(nodeaabb->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(primpos.data_ptr()),
      reinterpret_cast<const float*>(primrot.data_ptr()),
      reinterpret_cast<const float*>(primscale.data_ptr()),
      reinterpret_cast<float*>(raystepsim.data_ptr()),
      griddim,
      blocksizex,
      blocksizey,
      stream);

  return {};
}

std::vector<torch::Tensor> primintersection_backward(
    const torch::Tensor rayposim,
    const torch::Tensor raydirim,
    const torch::optional<torch::Tensor> sortedobjid,
    const torch::optional<torch::Tensor> nodechildren,
    const torch::optional<torch::Tensor> nodeaabb,
    const torch::Tensor primpos,
    const torch::Tensor primrot,
    const torch::Tensor primscale,
    const torch::Tensor raystepsim,
    const torch::Tensor grad_raysteps,
    torch::Tensor grad_primpos,
    torch::Tensor grad_primrot,
    torch::Tensor grad_primscale,
    int64_t griddim = 3,
    int64_t blocksizex = 256,
    int64_t blocksizey = 1) {
  CHECK_INPUT(rayposim);
  CHECK_INPUT(raydirim);
  if (sortedobjid) {
    CHECK_INPUT(*sortedobjid);
  }
  if (nodechildren) {
    CHECK_INPUT(*nodechildren);
  }
  if (nodeaabb) {
    CHECK_INPUT(*nodeaabb);
  }
  CHECK_INPUT(primpos);
  CHECK_INPUT(primrot);
  CHECK_INPUT(primscale);
  CHECK_INPUT(raystepsim);
  CHECK_INPUT(grad_raysteps);
  CHECK_INPUT(grad_primpos);
  CHECK_INPUT(grad_primrot);
  CHECK_INPUT(grad_primscale);

  const int N = rayposim.size(0);
  const int H = rayposim.size(1);
  const int W = rayposim.size(2);
  const int K = primpos.size(1);

  c10::cuda::CUDAGuard deviceGuard{rayposim.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  primintersection_backward_cuda(
      N,
      H,
      W,
      K,
      reinterpret_cast<const float*>(rayposim.data_ptr()),
      reinterpret_cast<const float*>(raydirim.data_ptr()),
      sortedobjid ? reinterpret_cast<const int*>(sortedobjid->data_ptr()) : nullptr,
      nodechildren ? reinterpret_cast<const int*>(nodechildren->data_ptr()) : nullptr,
      nodeaabb ? reinterpret_cast<const float*>(nodeaabb->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(primpos.data_ptr()),
      reinterpret_cast<const float*>(primrot.data_ptr()),
      reinterpret_cast<const float*>(primscale.data_ptr()),
      reinterpret_cast<const float*>(raystepsim.data_ptr()),
      reinterpret_cast<const float*>(grad_raysteps.data_ptr()),
      reinterpret_cast<float*>(grad_primpos.data_ptr()),
      reinterpret_cast<float*>(grad_primrot.data_ptr()),
      reinterpret_cast<float*>(grad_primscale.data_ptr()),
      griddim,
      blocksizex,
      blocksizey,
      stream);

  return {};
}

#ifndef NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("primintersection_forward", &primintersection_forward, "primintersection forward (CUDA)");
  m.def(
      "primintersection_backward", &primintersection_backward, "primintersection backward (CUDA)");
}
#endif

TORCH_LIBRARY(primintersection_ext, m) {
  m.def("primintersection_forward", &primintersection_forward);
  m.def("primintersection_backward", &primintersection_backward);
}
