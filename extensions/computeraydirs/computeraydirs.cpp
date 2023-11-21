#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <vector>

void compute_raydirs_forward_cuda(
    int N,
    int H,
    int W,
    float* viewpos,
    float* viewrot,
    float* focal,
    float* princpt,
    float* pixelcoordsim,
    float volradius,
    float* raypos,
    float* raydir,
    float* tminmax,
    cudaStream_t stream);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA((x));     \
  CHECK_CONTIGUOUS((x))

std::vector<torch::Tensor> compute_raydirs_forward(
    torch::Tensor viewpos,
    torch::Tensor viewrot,
    torch::Tensor focal,
    torch::Tensor princpt,
    torch::optional<torch::Tensor> pixelcoordsim,
    int64_t H,
    int64_t W,
    double volradius,
    torch::Tensor rayposim,
    torch::Tensor raydirim,
    torch::Tensor tminmaxim) {
  CHECK_INPUT(viewpos);
  CHECK_INPUT(viewrot);
  CHECK_INPUT(focal);
  CHECK_INPUT(princpt);
  if (pixelcoordsim) {
    CHECK_INPUT(*pixelcoordsim);
  }
  CHECK_INPUT(rayposim);
  CHECK_INPUT(raydirim);
  CHECK_INPUT(tminmaxim);

  c10::cuda::CUDAGuard deviceGuard{viewpos.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  int N = viewpos.size(0);
  assert(!pixelcoordsim || (pixelcoordsim->size(1) == H && pixelcoordsim->size(2) == W));

  compute_raydirs_forward_cuda(
      N,
      H,
      W,
      viewpos.data_ptr<float>(),
      viewrot.data_ptr<float>(),
      focal.data_ptr<float>(),
      princpt.data_ptr<float>(),
      pixelcoordsim ? pixelcoordsim->data_ptr<float>() : nullptr,
      volradius,
      rayposim.data_ptr<float>(),
      raydirim.data_ptr<float>(),
      tminmaxim.data_ptr<float>(),
      stream);

  return {};
}

#ifndef NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_raydirs_forward", &compute_raydirs_forward, "raydirs forward (CUDA)");
}
#endif

TORCH_LIBRARY(computeraydirs_ext, m) {
  m.def("compute_raydirs_forward", &compute_raydirs_forward);
}
