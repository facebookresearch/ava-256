#pragma once

#include <cmath>
#include <limits>

#include "../common/helper_math.h"

/*
 * Helper types to map from a scalar + dim -> packed CUDA vector types (float3, etc...)
 */
template <typename scalar_t, size_t dim>
struct vec_t {};
template <>
struct vec_t<float, 2> {
  using type = float2;

  template <typename... Args>
  static inline float2 make(Args&&... args) {
    return make_float2(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<int, 2> {
  using type = int2;

  template <typename... Args>
  static inline int2 make(Args&&... args) {
    return make_int2(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<uint8_t, 2> {
  using type = uchar2;

  template <typename... Args>
  static inline uchar2 make(Args&&... args) {
    return make_uchar2(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<float, 3> {
  using type = float3;

  template <typename... Args>
  static inline float3 make(Args&&... args) {
    return make_float3(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<int, 3> {
  using type = int3;

  template <typename... Args>
  static inline int3 make(Args&&... args) {
    return make_int3(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<uint8_t, 3> {
  using type = uchar3;

  template <typename... Args>
  static inline uchar3 make(Args&&... args) {
    return make_uchar3(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<float, 4> {
  using type = float4;

  template <typename... Args>
  static inline float4 make(Args&&... args) {
    return make_float4(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<int, 4> {
  using type = int4;

  template <typename... Args>
  static inline int4 make(Args&&... args) {
    return make_int4(std::forward<Args>(args)...);
  }
};
template <>
struct vec_t<uint8_t, 4> {
  using type = uchar4;

  template <typename... Args>
  static inline uchar4 make(Args&&... args) {
    return make_uchar4(std::forward<Args>(args)...);
  }
};

template <typename vec_t>
struct dim {};
template <>
struct dim<float2> {
  static constexpr int value = 2;
};
template <>
struct dim<float3> {
  static constexpr int value = 3;
};
template <>
struct dim<float4> {
  static constexpr int value = 4;
};
template <>
struct dim<uchar2> {
  static constexpr int value = 2;
};
template <>
struct dim<uchar3> {
  static constexpr int value = 3;
};
template <>
struct dim<uchar4> {
  static constexpr int value = 4;
};

static __forceinline__ __device__ float clock_diff(long long int end, long long int start) {
  long long int max_clock = std::numeric_limits<long long int>::max();
  return (end < start ? (end + float(max_clock - start)) : float(end - start));
}

static __forceinline__ __device__ bool allgt(float3 a, float3 b) {
  return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

static __forceinline__ __device__ bool alllt(float3 a, float3 b) {
  return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

static __forceinline__ __device__ float4 softplus(float4 x) {
  return make_float4(
      x.x > 20.f ? x.x : logf(1.f + expf(x.x)),
      x.y > 20.f ? x.y : logf(1.f + expf(x.y)),
      x.z > 20.f ? x.z : logf(1.f + expf(x.z)),
      x.w > 20.f ? x.w : logf(1.f + expf(x.w)));
}

static __forceinline__ __device__ float4 sigmoid(float4 x) {
  return make_float4(
      1.f / (1.f + expf(-x.x)),
      1.f / (1.f + expf(-x.y)),
      1.f / (1.f + expf(-x.z)),
      1.f / (1.f + expf(-x.w)));
}

static __forceinline__ __device__ bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

/*
 * Operators that convert uchar[34] -> float[34] implicitly.
 */

inline __host__ __device__ float3 operator+(uchar3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(uchar3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uchar3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, uchar3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3& a, uchar3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
inline __host__ __device__ float3 operator*(uchar3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(uchar3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uchar3& a, float3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, uchar3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3& a, uchar3 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}

inline __host__ __device__ float4 operator+(uchar4 a, float b) {
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(uchar4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(uchar4& a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, uchar4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4& a, uchar4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__ float4 operator*(uchar4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(uchar4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(uchar4& a, float4 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, uchar4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4& a, uchar4 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}

static __forceinline__ __device__ void safe_add_3d(
    float* data,
    int d,
    int h,
    int w,
    int sD,
    int sH,
    int sW,
    int D,
    int H,
    int W,
    float delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    atomicAdd(data + d * sD + h * sH + w * sW, delta);
  }
}

static __forceinline__ __device__ void safe_add_3d(
    float3* data,
    int d,
    int h,
    int w,
    int sD,
    int sH,
    int sW,
    int D,
    int H,
    int W,
    float3 delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 3 + 0, delta.x);
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 3 + 1, delta.y);
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 3 + 2, delta.z);
  }
}

static __forceinline__ __device__ void safe_add_3d(
    float4* data,
    int d,
    int h,
    int w,
    int sD,
    int sH,
    int sW,
    int D,
    int H,
    int W,
    float4 delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 0, delta.x);
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 1, delta.y);
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 2, delta.z);
    atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 3, delta.w);
  }
}

static __forceinline__ __device__ float clip_coordinates(float in, int clip_limit) {
  return ::min(static_cast<float>(clip_limit - 1), ::max(in, 0.f));
}

template <typename scalar_t>
static __forceinline__ __device__ float
clip_coordinates_set_grad(float in, int clip_limit, scalar_t* grad_in) {
  if (in < 0.f) {
    *grad_in = static_cast<scalar_t>(0);
    return 0.f;
  } else {
    float max = static_cast<float>(clip_limit - 1);
    if (in > max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

template <typename out_t, typename in_t>
static __device__ out_t grid_sample_forward(
    int C,
    int inp_D,
    int inp_H,
    int inp_W,
    const in_t* vals,
    float3 pos,
    bool border) {
  int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H, inp_sC = inp_W * inp_H * inp_D;
  int out_sC = 1;

  // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
  float ix = ((pos.x + 1.f) / 2) * (inp_W - 1);
  float iy = ((pos.y + 1.f) / 2) * (inp_H - 1);
  float iz = ((pos.z + 1.f) / 2) * (inp_D - 1);

  if (border) {
    // clip coordinates to image borders
    ix = clip_coordinates(ix, inp_W);
    iy = clip_coordinates(iy, inp_H);
    iz = clip_coordinates(iz, inp_D);
  }

  // get corner pixel values from (x, y, z)
  // for 4d, we used north-east-south-west
  // for 5d, we add top-bottom
  int ix_tnw = static_cast<int>(::floor(ix));
  int iy_tnw = static_cast<int>(::floor(iy));
  int iz_tnw = static_cast<int>(::floor(iz));

  int ix_tne = ix_tnw + 1;
  int iy_tne = iy_tnw;
  int iz_tne = iz_tnw;

  int ix_tsw = ix_tnw;
  int iy_tsw = iy_tnw + 1;
  int iz_tsw = iz_tnw;

  int ix_tse = ix_tnw + 1;
  int iy_tse = iy_tnw + 1;
  int iz_tse = iz_tnw;

  int ix_bnw = ix_tnw;
  int iy_bnw = iy_tnw;
  int iz_bnw = iz_tnw + 1;

  int ix_bne = ix_tnw + 1;
  int iy_bne = iy_tnw;
  int iz_bne = iz_tnw + 1;

  int ix_bsw = ix_tnw;
  int iy_bsw = iy_tnw + 1;
  int iz_bsw = iz_tnw + 1;

  int ix_bse = ix_tnw + 1;
  int iy_bse = iy_tnw + 1;
  int iz_bse = iz_tnw + 1;

  // get surfaces to each neighbor:
  float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
  float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
  float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
  float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
  float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
  float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
  float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
  float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

  out_t result;
  // auto inp_ptr_NC = input.data + n * inp_sN;
  // auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
  const in_t* inp_ptr_NC = vals;
  float* out_ptr_NCDHW = &result.x;
  for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
    //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
    // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
    // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
    // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
    *out_ptr_NCDHW = static_cast<float>(0);
    if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
    }
    if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
    }
    if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
    }
    if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
    }
    if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
    }
    if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
    }
    if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
    }
    if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
    }
  }
  return result;
}

// this dummy struct necessary because c++ is dumb
template <typename out_t, typename in_t>
struct GridSamplerForward {
  static __forceinline__ __device__ out_t
  exec(int C, int inp_D, int inp_H, int inp_W, const in_t* vals, float3 pos, bool border) {
    return grid_sample_forward<out_t, in_t>(C, inp_D, inp_H, inp_W, vals, pos, border);
  }
};

template <typename out_t, typename in_t>
static __device__ float3 grid_sample_backward(
    int C,
    int inp_D,
    int inp_H,
    int inp_W,
    const in_t* vals,
    in_t* grad_vals,
    float3 pos,
    out_t grad_out,
    bool border) {
  // int inp_W = D, inp_H = D, inp_D = D;
  int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H, inp_sC = inp_W * inp_H * inp_D;
  int gInp_sW = 1, gInp_sH = inp_W, gInp_sD = inp_W * inp_H, gInp_sC = inp_W * inp_H * inp_D;
  // int out_sC = 1;
  int gOut_sC = 1;

  // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
  float ix = ((pos.x + 1.f) / 2) * (inp_W - 1);
  float iy = ((pos.y + 1.f) / 2) * (inp_H - 1);
  float iz = ((pos.z + 1.f) / 2) * (inp_D - 1);

  // float gix_mult = static_cast<float>(1);
  // float giy_mult = static_cast<float>(1);
  // float giz_mult = static_cast<float>(1);
  float gix_mult = (inp_W - 1.f) / 2;
  float giy_mult = (inp_H - 1.f) / 2;
  float giz_mult = (inp_D - 1.f) / 2;

  if (border) {
    // clip coordinates to image borders
    ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
    iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
    iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
  }

  // get corner pixel values from (x, y, z)
  // for 4d, we used north-east-south-west
  // for 5d, we add top-bottom
  int ix_tnw = static_cast<int>(::floor(ix));
  int iy_tnw = static_cast<int>(::floor(iy));
  int iz_tnw = static_cast<int>(::floor(iz));

  int ix_tne = ix_tnw + 1;
  int iy_tne = iy_tnw;
  int iz_tne = iz_tnw;

  int ix_tsw = ix_tnw;
  int iy_tsw = iy_tnw + 1;
  int iz_tsw = iz_tnw;

  int ix_tse = ix_tnw + 1;
  int iy_tse = iy_tnw + 1;
  int iz_tse = iz_tnw;

  int ix_bnw = ix_tnw;
  int iy_bnw = iy_tnw;
  int iz_bnw = iz_tnw + 1;

  int ix_bne = ix_tnw + 1;
  int iy_bne = iy_tnw;
  int iz_bne = iz_tnw + 1;

  int ix_bsw = ix_tnw;
  int iy_bsw = iy_tnw + 1;
  int iz_bsw = iz_tnw + 1;

  int ix_bse = ix_tnw + 1;
  int iy_bse = iy_tnw + 1;
  int iz_bse = iz_tnw + 1;

  // get surfaces to each neighbor:
  float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
  float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
  float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
  float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
  float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
  float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
  float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
  float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

  float gix = static_cast<float>(0), giy = static_cast<float>(0), giz = static_cast<float>(0);
  // float *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w *
  // gOut_sW; float *gInp_ptr_NC = grad_input.data + n * gInp_sN; float *inp_ptr_NC = input.data + n
  // * inp_sN;
  float* gOut_ptr_NCDHW = &grad_out.x;
  in_t* gInp_ptr_NC = grad_vals;
  const in_t* inp_ptr_NC = vals;
  // calculate bilinear weighted pixel value and set output pixel
  for (int c = 0; c < C;
       ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
    float gOut = *gOut_ptr_NCDHW;

    // calculate and set grad_input
    safe_add_3d(
        gInp_ptr_NC,
        iz_tnw,
        iy_tnw,
        ix_tnw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tnw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_tne,
        iy_tne,
        ix_tne,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tne * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_tsw,
        iy_tsw,
        ix_tsw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tsw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_tse,
        iy_tse,
        ix_tse,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tse * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bnw,
        iy_bnw,
        ix_bnw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bnw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bne,
        iy_bne,
        ix_bne,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bne * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bsw,
        iy_bsw,
        ix_bsw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bsw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bse,
        iy_bse,
        ix_bse,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bse * gOut);

    // calculate grad_grid
    if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
      float tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
      gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
      giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
      giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
    }
    if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
      float tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
      gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
      giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
      giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
    }
    if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
      float tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
      gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
      giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
      giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
    }
    if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
      float tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
      gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
      giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
      giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
    }
    if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
      float bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
      gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
      giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
      giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
    }
    if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
      float bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
      gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
      giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
      giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
    }
    if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
      float bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
      gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
      giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
      giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
    }
    if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
      float bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
      gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
      giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
      giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
    }
  }

  // un-normalize grad_grid values back to [-1, 1] constraints
  // gix = gix * (inp_W - 1) / 2;
  // giy = giy * (inp_H - 1) / 2;
  // giz = giz * (inp_D - 1) / 2;

  // assuming grad_grid is contiguous
  // thus we can
  //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NDHW
  //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
  // scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
  // gGrid_ptr_NDHW[0] = gix_mult * gix;
  // gGrid_ptr_NDHW[1] = giy_mult * giy;
  // gGrid_ptr_NDHW[2] = giz_mult * giz;
  return make_float3(gix_mult * gix, giy_mult * giy, giz_mult * giz);
}

template <typename out_t, typename in_t>
struct GridSamplerBackward {
  static __forceinline__ __device__ float3 exec(
      int C,
      int inp_D,
      int inp_H,
      int inp_W,
      const in_t* vals,
      in_t* grad_vals,
      float3 pos,
      out_t grad_out,
      bool border) {
    return grid_sample_backward<out_t, in_t>(
        C, inp_D, inp_H, inp_W, vals, grad_vals, pos, grad_out, border);
  }
};

// template<class T>
// struct Zeros {
//    static __device__ T get();
//};
//
// template<>
// static __device__ float4 Zeros<float4>::get() {
//    return make_float4(0.f);
//}
//
// template<>
// static __device__ float3 Zeros<float3>::get() {
//    return make_float3(0.f);
//}

// template<class T>
// static __forceinline__ __device__ T getzeros() {
//    // no
//}
//
// template<>
// static __forceinline__ __device__ float4 getzeros() {
//    return make_float4(0.f);
//}
//
// template<>
// static __forceinline__ __device__ float3 getzeros() {
//    return make_float3(0.f);
//}

template <class out_t, class in_t>
static __device__ out_t grid_sample_chlast_forward(
    int,
    int inp_D,
    int inp_H,
    int inp_W,
    const in_t* vals,
    float3 pos,
    bool border) {
  int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H;

  // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
  float ix = ((pos.x + 1.f) / 2) * (inp_W - 1);
  float iy = ((pos.y + 1.f) / 2) * (inp_H - 1);
  float iz = ((pos.z + 1.f) / 2) * (inp_D - 1);

  if (border) {
    // clip coordinates to image borders
    ix = clip_coordinates(ix, inp_W);
    iy = clip_coordinates(iy, inp_H);
    iz = clip_coordinates(iz, inp_D);
  }

  // get corner pixel values from (x, y, z)
  // for 4d, we used north-east-south-west
  // for 5d, we add top-bottom
  int ix_tnw = static_cast<int>(::floor(ix));
  int iy_tnw = static_cast<int>(::floor(iy));
  int iz_tnw = static_cast<int>(::floor(iz));

  int ix_tne = ix_tnw + 1;
  int iy_tne = iy_tnw;
  int iz_tne = iz_tnw;

  int ix_tsw = ix_tnw;
  int iy_tsw = iy_tnw + 1;
  int iz_tsw = iz_tnw;

  int ix_tse = ix_tnw + 1;
  int iy_tse = iy_tnw + 1;
  int iz_tse = iz_tnw;

  int ix_bnw = ix_tnw;
  int iy_bnw = iy_tnw;
  int iz_bnw = iz_tnw + 1;

  int ix_bne = ix_tnw + 1;
  int iy_bne = iy_tnw;
  int iz_bne = iz_tnw + 1;

  int ix_bsw = ix_tnw;
  int iy_bsw = iy_tnw + 1;
  int iz_bsw = iz_tnw + 1;

  int ix_bse = ix_tnw + 1;
  int iy_bse = iy_tnw + 1;
  int iz_bse = iz_tnw + 1;

  // get surfaces to each neighbor:
  float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
  float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
  float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
  float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
  float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
  float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
  float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
  float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

  // out_t result = Zeros<out_t>::get();
  out_t result;
  memset(&result, 0, sizeof(out_t));

  auto inp_ptr_NC = (const typename vec_t<in_t, dim<out_t>::value>::type*)vals;
  out_t* out_ptr_NCDHW = &result;
  {
    if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
    }
    if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
    }
    if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
    }
    if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
    }
    if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
    }
    if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
    }
    if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
    }
    if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
      *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
    }
  }
  return result;
}

template <typename out_t, typename in_t>
struct GridSamplerChlastForward {
  static __forceinline__ __device__ out_t
  exec(int C, int inp_D, int inp_H, int inp_W, const in_t* vals, float3 pos, bool border) {
    return grid_sample_chlast_forward<out_t, in_t>(C, inp_D, inp_H, inp_W, vals, pos, border);
  }
};

template <typename out_t, typename in_t>
static __device__ float3 grid_sample_chlast_backward(
    int,
    int inp_D,
    int inp_H,
    int inp_W,
    const in_t* vals,
    in_t* grad_vals,
    float3 pos,
    out_t grad_out,
    bool border) {
  int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H; //, inp_sC = inp_W * inp_H * inp_D;
  int gInp_sW = 1, gInp_sH = inp_W, gInp_sD = inp_W * inp_H; //, gInp_sC = inp_W * inp_H * inp_D;
  // int gOut_sC = 1;

  // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
  float ix = ((pos.x + 1.f) / 2) * (inp_W - 1);
  float iy = ((pos.y + 1.f) / 2) * (inp_H - 1);
  float iz = ((pos.z + 1.f) / 2) * (inp_D - 1);

  float gix_mult = (inp_W - 1.f) / 2;
  float giy_mult = (inp_H - 1.f) / 2;
  float giz_mult = (inp_D - 1.f) / 2;

  if (border) {
    // clip coordinates to image borders
    ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
    iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
    iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
  }

  // get corner pixel values from (x, y, z)
  // for 4d, we used north-east-south-west
  // for 5d, we add top-bottom
  int ix_tnw = static_cast<int>(::floor(ix));
  int iy_tnw = static_cast<int>(::floor(iy));
  int iz_tnw = static_cast<int>(::floor(iz));

  int ix_tne = ix_tnw + 1;
  int iy_tne = iy_tnw;
  int iz_tne = iz_tnw;

  int ix_tsw = ix_tnw;
  int iy_tsw = iy_tnw + 1;
  int iz_tsw = iz_tnw;

  int ix_tse = ix_tnw + 1;
  int iy_tse = iy_tnw + 1;
  int iz_tse = iz_tnw;

  int ix_bnw = ix_tnw;
  int iy_bnw = iy_tnw;
  int iz_bnw = iz_tnw + 1;

  int ix_bne = ix_tnw + 1;
  int iy_bne = iy_tnw;
  int iz_bne = iz_tnw + 1;

  int ix_bsw = ix_tnw;
  int iy_bsw = iy_tnw + 1;
  int iz_bsw = iz_tnw + 1;

  int ix_bse = ix_tnw + 1;
  int iy_bse = iy_tnw + 1;
  int iz_bse = iz_tnw + 1;

  // get surfaces to each neighbor:
  float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
  float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
  float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
  float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
  float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
  float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
  float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
  float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

  float gix = static_cast<float>(0), giy = static_cast<float>(0), giz = static_cast<float>(0);
  out_t* gOut_ptr_NCDHW = &grad_out;

  auto inp_ptr_NC = (const typename vec_t<in_t, dim<out_t>::value>::type*)vals;
  auto gInp_ptr_NC = (typename vec_t<in_t, dim<out_t>::value>::type*)grad_vals;

  // calculate bilinear weighted pixel value and set output pixel
  // for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC +=
  // inp_sC) {
  {
    out_t gOut = *gOut_ptr_NCDHW;

    // calculate and set grad_input
    safe_add_3d(
        gInp_ptr_NC,
        iz_tnw,
        iy_tnw,
        ix_tnw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tnw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_tne,
        iy_tne,
        ix_tne,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tne * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_tsw,
        iy_tsw,
        ix_tsw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tsw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_tse,
        iy_tse,
        ix_tse,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        tse * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bnw,
        iy_bnw,
        ix_bnw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bnw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bne,
        iy_bne,
        ix_bne,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bne * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bsw,
        iy_bsw,
        ix_bsw,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bsw * gOut);
    safe_add_3d(
        gInp_ptr_NC,
        iz_bse,
        iy_bse,
        ix_bse,
        gInp_sD,
        gInp_sH,
        gInp_sW,
        inp_D,
        inp_H,
        inp_W,
        bse * gOut);

    // calculate grad_grid
    if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
      out_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
      gix -= (iy_bse - iy) * (iz_bse - iz) * dot(tnw_val, gOut);
      giy -= (ix_bse - ix) * (iz_bse - iz) * dot(tnw_val, gOut);
      giz -= (ix_bse - ix) * (iy_bse - iy) * dot(tnw_val, gOut);
    }
    if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
      out_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
      gix += (iy_bsw - iy) * (iz_bsw - iz) * dot(tne_val, gOut);
      giy -= (ix - ix_bsw) * (iz_bsw - iz) * dot(tne_val, gOut);
      giz -= (ix - ix_bsw) * (iy_bsw - iy) * dot(tne_val, gOut);
    }
    if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
      out_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
      gix -= (iy - iy_bne) * (iz_bne - iz) * dot(tsw_val, gOut);
      giy += (ix_bne - ix) * (iz_bne - iz) * dot(tsw_val, gOut);
      giz -= (ix_bne - ix) * (iy - iy_bne) * dot(tsw_val, gOut);
    }
    if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
      out_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
      gix += (iy - iy_bnw) * (iz_bnw - iz) * dot(tse_val, gOut);
      giy += (ix - ix_bnw) * (iz_bnw - iz) * dot(tse_val, gOut);
      giz -= (ix - ix_bnw) * (iy - iy_bnw) * dot(tse_val, gOut);
    }
    if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
      out_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
      gix -= (iy_tse - iy) * (iz - iz_tse) * dot(bnw_val, gOut);
      giy -= (ix_tse - ix) * (iz - iz_tse) * dot(bnw_val, gOut);
      giz += (ix_tse - ix) * (iy_tse - iy) * dot(bnw_val, gOut);
    }
    if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
      out_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
      gix += (iy_tsw - iy) * (iz - iz_tsw) * dot(bne_val, gOut);
      giy -= (ix - ix_tsw) * (iz - iz_tsw) * dot(bne_val, gOut);
      giz += (ix - ix_tsw) * (iy_tsw - iy) * dot(bne_val, gOut);
    }
    if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
      out_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
      gix -= (iy - iy_tne) * (iz - iz_tne) * dot(bsw_val, gOut);
      giy += (ix_tne - ix) * (iz - iz_tne) * dot(bsw_val, gOut);
      giz += (ix_tne - ix) * (iy - iy_tne) * dot(bsw_val, gOut);
    }
    if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
      out_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
      gix += (iy - iy_tnw) * (iz - iz_tnw) * dot(bse_val, gOut);
      giy += (ix - ix_tnw) * (iz - iz_tnw) * dot(bse_val, gOut);
      giz += (ix - ix_tnw) * (iy - iy_tnw) * dot(bse_val, gOut);
    }
  }

  return make_float3(gix_mult * gix, giy_mult * giy, giz_mult * giz);
}

template <typename out_t, typename in_t>
struct GridSamplerChlastBackward {
  static __forceinline__ __device__ float3 exec(
      int C,
      int inp_D,
      int inp_H,
      int inp_W,
      const in_t* vals,
      in_t* grad_vals,
      float3 pos,
      out_t grad_out,
      bool border) {
    return grid_sample_chlast_backward<out_t, in_t>(
        C, inp_D, inp_H, inp_W, vals, grad_vals, pos, grad_out, border);
  }
};

inline __host__ __device__ float min_component(float3 a) {
  return fminf(fminf(a.x, a.y), a.z);
}

inline __host__ __device__ float max_component(float3 a) {
  return fmaxf(fmaxf(a.x, a.y), a.z);
}

inline __host__ __device__ float3 abs(float3 a) {
  return make_float3(abs(a.x), abs(a.y), abs(a.z));
}

// A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering
// http://jcgt.org/published/0007/03/04/
__forceinline__ __device__ bool ray_aabb_hit(float3 p0, float3 p1, float3 raypos, float3 raydir) {
  float3 t0 = (p0 - raypos) / raydir;
  float3 t1 = (p1 - raypos) / raydir;
  float3 tmin = fminf(t0, t1), tmax = fmaxf(t0, t1);

  return max_component(tmin) <= min_component(tmax);
}
__forceinline__ __device__ bool ray_aabb_hit_ird(float3 p0, float3 p1, float3 raypos, float3 ird) {
  float3 t0 = (p0 - raypos) * ird;
  float3 t1 = (p1 - raypos) * ird;
  float3 tmin = fminf(t0, t1), tmax = fmaxf(t0, t1);

  return max_component(tmin) <= min_component(tmax);
}
__forceinline__ __device__ void ray_aabb_hit_ird_tminmax(
    float3 p0,
    float3 p1,
    float3 raypos,
    float3 ird,
    float& otmin,
    float& otmax) {
  float3 t0 = (p0 - raypos) * ird;
  float3 t1 = (p1 - raypos) * ird;
  float3 tmin = fminf(t0, t1), tmax = fmaxf(t0, t1);
  tmin = fminf(t0, t1);
  tmax = fmaxf(t0, t1);
  otmin = max_component(tmin);
  otmax = min_component(tmax);
}

inline __device__ bool
aabb_intersect(float3 p0, float3 p1, float3 r0, float3 rd, float& tmin, float& tmax) {
  //   https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

  // BUGGY but illustrative
  float tymin, tymax, tzmin, tzmax;
  const float3 bounds[2] = {p0, p1};
  float3 ird = 1.0f / rd;
  int sx = (ird.x < 0) ? 1 : 0;
  int sy = (ird.y < 0) ? 1 : 0;
  int sz = (ird.z < 0) ? 1 : 0;
  tmin = (bounds[sx].x - r0.x) * ird.x;
  tmax = (bounds[1 - sx].x - r0.x) * ird.x;
  tymin = (bounds[sy].y - r0.y) * ird.y;
  tymax = (bounds[1 - sy].y - r0.y) * ird.y;

  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin)
    tmin = tymin;
  if (tymax < tmax)
    tmax = tymax;

  tzmin = (bounds[sz].z - r0.z) * ird.z;
  tzmax = (bounds[1 - sz].z - r0.z) * ird.z;

  if ((tmin > tzmax) || (tzmin > tmax))
    return false;
  if (tzmin > tmin)
    tmin = tzmin;
  if (tzmax < tmax)
    tmax = tzmax;

  return true;
}

template <int maxhitboxes>
static __device__ void ray_bvh_tminmax_subset(
    unsigned warpmask,
    int K,
    int n,
    const int* sortedobjid,
    const int2* nodechildren,
    const float3* nodeaabb,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3 raypos,
    const float3 raydir,
    const float2 tminmax,
    float2& rtminmax,
    int* hitboxes,
    int& num) {
  float3 iraydir = 1.0f / raydir;
  int stack[64];
  int* stack_ptr = stack;
  *stack_ptr++ = -1;
  int node = 0;
  do {
    int2 children = nodechildren[n * (K + K - 1) + node];

    // check if we're in a leaf
    if (children.x < 0) {
      int k0 = -children.x - 1;

      {
        int ks = k0;
        int k = sortedobjid[n * K + ks];
        float3 w1t = primpos[n * K + k];
        float3 w1r0 = primrot[(n * K + k) * 3 + 0];
        float3 w1r1 = primrot[(n * K + k) * 3 + 1];
        float3 w1r2 = primrot[(n * K + k) * 3 + 2];
        float3 w1s = primscale[n * K + k];

        // Ray origin in box coordinates
        float3 r0 = raypos - w1t;
        r0 = (w1r0 * r0.x + w1r1 * r0.y + w1r2 * r0.z);
        // Ray direction in box coordinates
        float3 rd = (w1r0 * raydir.x + w1r1 * raydir.y + w1r2 * raydir.z);

        // Intersect with box
        float3 ird = 1.0f / rd;
        float3 iw1s = 1.0f / w1s;
        float3 t0 = (-iw1s - r0) * ird;
        float3 t1 = (iw1s - r0) * ird;
        float3 tmin = fminf(t0, t1), tmax = fmaxf(t0, t1);

        float trmin = max_component(tmin);
        float trmax = min_component(tmax);

        // Crop to original segment
        trmin = fmaxf(tminmax.x, trmin);
        trmax = fminf(tminmax.y, trmax);

        if (trmin <= trmax) {
          // hit
          rtminmax.x = fminf(rtminmax.x, trmin);
          rtminmax.y = fmaxf(rtminmax.y, trmax);

          if (num < maxhitboxes) {
            hitboxes[num++] = k;
          }
        }
      }

      node = *--stack_ptr;
    } else {
      // check if we're in each child's bbox
      const float3* nodeaabb_ptr = nodeaabb + n * (K + K - 1) * 2 + children.x * 2;
      bool traverse_l = ray_aabb_hit_ird(nodeaabb_ptr[0], nodeaabb_ptr[1], raypos, iraydir);
      nodeaabb_ptr = nodeaabb + n * (K + K - 1) * 2 + children.y * 2;
      bool traverse_r = ray_aabb_hit_ird(nodeaabb_ptr[0], nodeaabb_ptr[1], raypos, iraydir);

      // update stack
      if (!traverse_l && !traverse_r) {
        node = *--stack_ptr;
      } else {
        node = traverse_l ? children.x : children.y;
        if (traverse_l && traverse_r) {
          *stack_ptr++ = children.y;
        }
      }
    }
  } while (node != -1);
}

template <int maxhitboxes>
struct RayBVHSubset {
  static __forceinline__ __device__ void exec(
      unsigned warpmask,
      int K,
      int n,
      const int* sortedobjid,
      const int2* nodechildren,
      const float3* nodeaabb,
      const float3* primpos,
      const float3* primrot,
      const float3* primscale,
      const float3 raypos,
      const float3 raydir,
      const float2 tminmax,
      float2& rtminmax,
      int* hitboxes,
      int& num) {
    return ray_bvh_tminmax_subset<maxhitboxes>(
        warpmask,
        K,
        n,
        sortedobjid,
        nodechildren,
        nodeaabb,
        primpos,
        primrot,
        primscale,
        raypos,
        raydir,
        tminmax,
        rtminmax,
        hitboxes,
        num);
  }
};

template <int maxhitboxes>
static __forceinline__ __device__ void ray_bvh_tminmax_subset_sync(
    unsigned warpmask,
    int K,
    int n,
    const int* sortedobjid,
    const int2* nodechildren,
    const float3* nodeaabb,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3 raypos,
    const float3 raydir,
    const float2 tminmax,
    float2& rtminmax,
    int* hitboxes,
    int& num) {
  float3 iraydir = 1.0f / raydir;
  int stack[64];
  int* stack_ptr = stack;
  *stack_ptr++ = -1;
  int node = 0;
  do {
    int2 children = nodechildren[n * (K + K - 1) + node];

    // check if we're in a leaf
    if (__any_sync(warpmask, children.x < 0)) {
      int k0 = -children.x - 1;

      {
        int ks = k0;
        int k = sortedobjid[n * K + ks];

        float3 w1t = primpos[n * K + k];
        float3 w1r0 = primrot[(n * K + k) * 3 + 0];
        float3 w1r1 = primrot[(n * K + k) * 3 + 1];
        float3 w1r2 = primrot[(n * K + k) * 3 + 2];
        float3 w1s = primscale[n * K + k];

        // Ray origin in box coordinates
        float3 r0 = raypos - w1t;
        r0 = (w1r0 * r0.x + w1r1 * r0.y + w1r2 * r0.z);

        // Ray direction in box coordinates
        float3 rd = (w1r0 * raydir.x + w1r1 * raydir.y + w1r2 * raydir.z);

        // Intersect with box
        float3 ird = 1.0f / rd;
        float3 iw1s = 1.0f / w1s;
        float3 t0 = (-iw1s - r0) * ird;
        float3 t1 = (iw1s - r0) * ird;
        float3 tmin = fminf(t0, t1), tmax = fmaxf(t0, t1);

        float trmin = max_component(tmin);
        float trmax = min_component(tmax);

        // Crop to original segment
        trmin = fmaxf(tminmax.x, trmin);
        trmax = fminf(tminmax.y, trmax);

        if (trmin <= trmax) {
          // hit
          rtminmax.x = fminf(rtminmax.x, trmin);
          rtminmax.y = fmaxf(rtminmax.y, trmax);
        }

        if (__any_sync(warpmask, trmin <= trmax)) {
          if (num < maxhitboxes) {
            hitboxes[num++] = k;
          }
        }
      }

      node = *--stack_ptr;
    } else {
      // check if we're in each child's bbox
      const float3* nodeaabb_ptr = nodeaabb + n * (K + K - 1) * 2 + children.x * 2;
      bool traverse_l = ray_aabb_hit_ird(nodeaabb_ptr[0], nodeaabb_ptr[1], raypos, iraydir);
      nodeaabb_ptr = nodeaabb + n * (K + K - 1) * 2 + children.y * 2;
      bool traverse_r = ray_aabb_hit_ird(nodeaabb_ptr[0], nodeaabb_ptr[1], raypos, iraydir);

      traverse_l = __any_sync(warpmask, traverse_l);
      traverse_r = __any_sync(warpmask, traverse_r);

      // update stack
      if (!traverse_l && !traverse_r) {
        node = *--stack_ptr;
      } else {
        node = traverse_l ? children.x : children.y;
        if (traverse_l && traverse_r) {
          *stack_ptr++ = children.y;
        }
      }
    }
  } while (node != -1);
}

template <int maxhitboxes>
struct RayBVHSubsetSync {
  static __forceinline__ __device__ void exec(
      unsigned warpmask,
      int K,
      int n,
      const int* sortedobjid,
      const int2* nodechildren,
      const float3* nodeaabb,
      const float3* primpos,
      const float3* primrot,
      const float3* primscale,
      const float3 raypos,
      const float3 raydir,
      const float2 tminmax,
      float2& rtminmax,
      int* hitboxes,
      int& num) {
    return ray_bvh_tminmax_subset_sync<maxhitboxes>(
        warpmask,
        K,
        n,
        sortedobjid,
        nodechildren,
        nodeaabb,
        primpos,
        primrot,
        primscale,
        raypos,
        raydir,
        tminmax,
        rtminmax,
        hitboxes,
        num);
  }
};
