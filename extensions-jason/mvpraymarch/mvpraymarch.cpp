#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../common/utils.h"

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

void compute_morton_cuda(int N, int K, float* center, int* code, cudaStream_t stream);

void build_tree_cuda(
    int N,
    int K,
    int* sortedcode,
    int* nodechildren,
    int* nodeparent,
    cudaStream_t stream);

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
    cudaStream_t stream);

std::vector<torch::Tensor> compute_morton(torch::Tensor center, torch::Tensor code) {
  CHECK_INPUT(center);
  CHECK_INPUT(code);

  int N = center.size(0);
  int K = center.size(1);

  c10::cuda::CUDAGuard deviceGuard{center.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  compute_morton_cuda(
      N,
      K,
      reinterpret_cast<float*>(center.data_ptr()),
      reinterpret_cast<int*>(code.data_ptr()),
      stream);

  return {};
}

std::vector<torch::Tensor>
build_tree(torch::Tensor sortedcode, torch::Tensor nodechildren, torch::Tensor nodeparent) {
  CHECK_INPUT(sortedcode);
  CHECK_INPUT(nodechildren);
  CHECK_INPUT(nodeparent);

  int N = sortedcode.size(0);
  int K = sortedcode.size(1);

  c10::cuda::CUDAGuard deviceGuard{sortedcode.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  build_tree_cuda(
      N,
      K,
      reinterpret_cast<int*>(sortedcode.data_ptr()),
      reinterpret_cast<int*>(nodechildren.data_ptr()),
      reinterpret_cast<int*>(nodeparent.data_ptr()),
      stream);

  return {};
}

std::vector<torch::Tensor> compute_aabb2(
    torch::Tensor sortedobjid,
    torch::Tensor primpos,
    torch::Tensor primrot,
    torch::Tensor primscale,
    torch::Tensor nodechildren,
    torch::Tensor nodeparent,
    torch::Tensor nodeaabb) {
  CHECK_INPUT(sortedobjid);
  CHECK_INPUT(primpos);
  CHECK_INPUT(primrot);
  CHECK_INPUT(primscale);
  CHECK_INPUT(nodechildren);
  CHECK_INPUT(nodeparent);
  CHECK_INPUT(nodeaabb);

  int N = primpos.size(0);
  int K = primpos.size(1);

  c10::cuda::CUDAGuard deviceGuard{primpos.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  torch::TensorOptions options(primpos.device());
  auto atom = torch::zeros({N * (K - 1)}, options.dtype(torch::kInt32));

  compute_aabb2_cuda(
      N,
      K,
      sortedobjid.data_ptr<int>(),
      primpos.data_ptr<float>(),
      primrot.data_ptr<float>(),
      primscale.data_ptr<float>(),
      nodechildren.data_ptr<int>(),
      nodeparent.data_ptr<int>(),
      nodeaabb.data_ptr<float>(),
      atom.data_ptr<int>(),
      stream);

  return {};
}

std::vector<torch::Tensor> raymarch_forward(
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
    torch::Tensor rayrgbaim,
    torch::optional<torch::Tensor> raysatim,
    torch::optional<torch::Tensor> raytermim,
    torch::optional<torch::Tensor> t_im,
    bool chlast = false,
    double fadescale = 8.f,
    double fadeexp = 8.f,
    int64_t accum = 0,
    double termthresh = 0.f,
    int64_t griddim = 3,
    int64_t blocksizex = 6,
    int64_t blocksizey = 32) {
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
  if (t_im) {
    CHECK_INPUT(*t_im);
  }

  const int N = rayposim.size(0);
  const int H = rayposim.size(1);
  const int W = rayposim.size(2);
  const int K = primpos.size(1);
  const int MD = tplate.size(chlast ? 2 : 3);
  const int MH = tplate.size(chlast ? 3 : 4);
  const int MW = tplate.size(chlast ? 4 : 5);

  int TN = tplate.size(0);
  if (TN != N && TN != 1) {
    throw std::runtime_error("Template batch size must either match other batch sizes, or be 1.");
  }

  if (primpos.size(0) != TN || primrot.size(0) != TN || primscale.size(0) != TN) {
    throw std::runtime_error(
        "All slab components (template, primpos, primrot, primscale) must have the same batchsize.");
  }

  bool broadcast_slab = (TN != N);

  c10::cuda::CUDAGuard deviceGuard{rayposim.device()};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  if (tplate.dtype() == torch::kFloat32) {
    if (raysatim) {
      CHECK_INPUT(*raysatim);
    }
    if (raytermim) {
      CHECK_INPUT(*raytermim);
    }
    raymarch_forward_cuda_float(
        N,
        H,
        W,
        K,
        MD,
        MH,
        MW,
        rayposim.data_ptr<float>(),
        raydirim.data_ptr<float>(),
        stepsize,
        tminmaxim.data_ptr<float>(),
        sortedobjid ? sortedobjid->data_ptr<int>() : nullptr,
        nodechildren ? nodechildren->data_ptr<int>() : nullptr,
        nodeaabb ? nodeaabb->data_ptr<float>() : nullptr,
        tplate.data_ptr<float>(),
        warp ? warp->data_ptr<float>() : nullptr,
        primpos.data_ptr<float>(),
        primrot.data_ptr<float>(),
        primscale.data_ptr<float>(),
        rayrgbaim.data_ptr<float>(),
        raysatim ? raysatim->data_ptr<float>() : nullptr,
        raytermim ? raytermim->data_ptr<int>() : nullptr,
        t_im ? t_im->data_ptr<float>() : nullptr,
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
  } else if (tplate.dtype() == torch::kUInt8) {
    if (rayrgbaim.dtype() != torch::kUInt8) {
      throw std::runtime_error(
          "raymarch_forward, when given uint8 template, expects uint8 rayrgbaim as output.");
    }
    raymarch_forward_cuda_uint(
        N,
        H,
        W,
        K,
        MD,
        MH,
        MW,
        rayposim.data_ptr<float>(),
        raydirim.data_ptr<float>(),
        stepsize,
        tminmaxim.data_ptr<float>(),
        sortedobjid ? sortedobjid->data_ptr<int>() : nullptr,
        nodechildren ? nodechildren->data_ptr<int>() : nullptr,
        nodeaabb ? nodeaabb->data_ptr<float>() : nullptr,
        tplate.data_ptr<uint8_t>(),
        warp ? warp->data_ptr<float>() : nullptr,
        primpos.data_ptr<float>(),
        primrot.data_ptr<float>(),
        primscale.data_ptr<float>(),
        rayrgbaim.data_ptr<uint8_t>(),
        t_im ? t_im->data_ptr<float>() : nullptr,
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

  return {};
}

std::vector<torch::Tensor> raymarch_backward(
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
    const torch::optional<torch::Tensor> raysatim,
    const torch::optional<torch::Tensor> raytermim,
    const torch::Tensor grad_rayrgba,
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
    int64_t blocksizex = 6,
    int64_t blocksizey = 32) {
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
  if (raysatim) {
    CHECK_INPUT(*raysatim);
  }
  if (raytermim) {
    CHECK_INPUT(*raytermim);
  }
  CHECK_INPUT(grad_rayrgba);
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

  raymarch_backward_cuda(
      N,
      H,
      W,
      K,
      MD,
      MH,
      MW,
      reinterpret_cast<float*>(rayposim.data_ptr()),
      reinterpret_cast<float*>(raydirim.data_ptr()),
      stepsize,
      reinterpret_cast<float*>(tminmaxim.data_ptr()),
      sortedobjid ? reinterpret_cast<int*>(sortedobjid->data_ptr()) : nullptr,
      nodechildren ? reinterpret_cast<int*>(nodechildren->data_ptr()) : nullptr,
      nodeaabb ? reinterpret_cast<float*>(nodeaabb->data_ptr()) : nullptr,
      reinterpret_cast<float*>(tplate.data_ptr()),
      warp ? reinterpret_cast<float*>(warp->data_ptr()) : nullptr,
      reinterpret_cast<float*>(primpos.data_ptr()),
      reinterpret_cast<float*>(primrot.data_ptr()),
      reinterpret_cast<float*>(primscale.data_ptr()),
      reinterpret_cast<float*>(rayrgbaim.data_ptr()),
      raysatim ? reinterpret_cast<float*>(raysatim->data_ptr()) : nullptr,
      raytermim ? reinterpret_cast<int*>(raytermim->data_ptr()) : nullptr,
      reinterpret_cast<float*>(grad_rayrgba.data_ptr()),
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
  m.def("compute_morton", &compute_morton, "compute morton codes (CUDA)");
  m.def("build_tree", &build_tree, "build BVH tree (CUDA)");
  m.def("compute_aabb2", &compute_aabb2, "compute AABB sizes (CUDA)");

  m.def("raymarch_forward", &raymarch_forward, "raymarch forward (CUDA)");
  m.def("raymarch_backward", &raymarch_backward, "raymarch backward (CUDA)");
}
#endif

TORCH_LIBRARY(mvpraymarch_ext, m) {
  m.def("compute_morton", &compute_morton);
  m.def("build_tree", &build_tree);
  m.def("compute_aabb2", &compute_aabb2);

  m.def("raymarch_forward", &raymarch_forward);
  m.def("raymarch_backward", &raymarch_backward);
}
