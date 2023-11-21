#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>

#include <limits>

#include "utils.h"
#include "../common/helper_math.h"

// This function is a hacky gamma curve
constexpr float3 black3{3.f, 3.f, 3.f};
constexpr float3 zero3{0.f, 0.f, 0.f};
constexpr float hack_black = 15.f;
constexpr float _gamma = 1.8f;
constexpr float _igamma = 1.f / _gamma;

static __forceinline__ __device__ float3
output_color_curve(float3 rgb, const float igamma = _igamma, const float3 black = zero3) {
  const float3 rgbmul = 255.f * make_float3(1.273f, 1.f, 1.455f) / (255.f - black);
  rgb = clamp(
      255.f * pow(fmaxf(zero3, rgbmul * (rgb - black) / 255.f), make_float3(igamma)) - hack_black,
      0.f,
      255.f);
  return rgb;
}

static __forceinline__ __device__ float srgbcurve(float v) {
  return (v <= 0.0031308f) ? (v * 12.92f) : (1.055f * pow(v, 1.f / 2.4f) - 0.055f);
}

// This function implements the sRGB transfer function plus a white balance scaling
constexpr float3 rgbmul{1.33f / 255.f, 1.f / 255.f, 1.5f / 255.f};
static __forceinline__ __device__ float3
output_color_curve2(float3 rgb, const float igamma_igamma, const float3 black = zero3) {
  rgb = clamp(rgbmul * rgb, 0.f, 1.f);
  rgb.x = 255.f * srgbcurve(rgb.x);
  rgb.y = 255.f * srgbcurve(rgb.y);
  rgb.z = 255.f * srgbcurve(rgb.z);
  return rgb;
}

// This funtion does gamma decompression for uint8 templates
// Gamma should be matched to the value used in export.py
static __forceinline__ __device__ float3 tplate_ungamma(float3 rgb, const float gamma = _gamma) {
  rgb.x = 255.f * pow(max(0.f, rgb.x / 255.f), gamma);
  rgb.y = 255.f * pow(max(0.f, rgb.y / 255.f), gamma);
  rgb.z = 255.f * pow(max(0.f, rgb.z / 255.f), gamma);
  return rgb;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_forward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float t,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    float3* raysatim,
    float& t_acc,
    float fadescale,
    float fadeexp) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 y0 = raypos - w1t;
  y0 = (w1r0 * y0.x + w1r1 * y0.y + w1r2 * y0.z) * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    if (std::is_integral<tplate_t>::value) {
      rgb = tplate_ungamma(rgb);
    }
    float alpha = sample.w * fade;

    // accumulate
    float newalpha = rayrgba.w + alpha * stepsize;
    float contrib = fminf(newalpha, 1.f) - rayrgba.w;

    rayrgba += make_float4(rgb, 1.f) * contrib;
    t_acc += t * contrib;

    if (newalpha >= 1.f) {
      // save saturation point
      if (raysatim) {
        *raysatim = rgb;
      }
      return 1;
    }
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_backward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    const float3& raysat,
    float fadescale,
    float fadeexp,
    tplate_t* grad_tplate,
    float* grad_warp,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale,
    float4& dL_rayrgba) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 xmt1 = raypos - w1t;
  float3 rxmt1 = w1r0 * xmt1.x + w1r1 * xmt1.y + w1r2 * xmt1.z;
  float3 y0 = rxmt1 * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));
    float3 dfade_y0 = fade * -(fadescale * fadeexp) *
        make_float3(pow(abs(y0.x), fadeexp - 1.f) * (y0.x > 0.f ? 1.f : -1.f),
                    pow(abs(y0.y), fadeexp - 1.f) * (y0.y > 0.f ? 1.f : -1.f),
                    pow(abs(y0.z), fadeexp - 1.f) * (y0.z > 0.f ? 1.f : -1.f));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    float alpha = sample.w * fade;

    // accumulate
    float newalpha = rayrgba.w + alpha * stepsize;
    float contrib = fminf(newalpha, 1.f) - rayrgba.w;

    float rayrgbaprev = rayrgba.w;

    rayrgba += make_float4(rgb, 1.f) * contrib;

    float3 dL_rgb = newalpha >= 1.f ? (1.f - rayrgbaprev) * make_float3(dL_rayrgba)
                                    : contrib * make_float3(dL_rayrgba);
    float dL_contrib =
        dot(make_float4(rgb, 1.f) - (raysat.x > -1.f ? make_float4(raysat, 1.f) : make_float4(0.f)),
            dL_rayrgba);

    float dL_newalpha = newalpha >= 1.f ? 0.f : dL_contrib;
    float dL_alpha = stepsize * dL_newalpha;
    float4 dL_sample = make_float4(dL_rgb, fade * dL_alpha);

    float3 dL_y0;
    if (do_warp) {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      float3 dL_y1 = GridSamplerBackwardT<float4, tplate_t>::exec(
          4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y1, dL_sample, false);

      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      float* grad_warp_ptr = grad_warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float3, float>::exec(
                  3, MD, MH, MW, warp_ptr, grad_warp_ptr, y0, dL_y1, false) +
          dfade_y0 * sample.w * dL_alpha;
    } else {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float4, tplate_t>::exec(
                  4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y0, dL_sample, false) +
          dfade_y0 * sample.w * dL_alpha;
    }

    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 0, rxmt1.x * dL_y0.x);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 1, rxmt1.y * dL_y0.y);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 2, rxmt1.z * dL_y0.z);

    dL_y0 *= w1s;

    float3 gw1r0 = xmt1.x * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 0, gw1r0.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 1, gw1r0.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 2, gw1r0.z);

    float3 gw1r1 = xmt1.y * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 0, gw1r1.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 1, gw1r1.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 2, gw1r1.z);

    float3 gw1r2 = xmt1.z * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 0, gw1r2.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 1, gw1r2.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 2, gw1r2.z);

    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 0, -dot(w1r0, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 1, -dot(w1r1, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 2, -dot(w1r2, dL_y0));

    if (newalpha >= 1.f) {
      // save saturation point
      return 1;
    }
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_mult_forward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float t,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    float3* raysatim,
    float& t_acc,
    float fadescale,
    float fadeexp) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 y0 = raypos - w1t;
  y0 = (w1r0 * y0.x + w1r1 * y0.y + w1r2 * y0.z) * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    if (std::is_integral<tplate_t>::value) {
      rgb = tplate_ungamma(rgb);
    }
    float alpha = sample.w * fade * stepsize;

    // accumulate
    float contrib = exp(-rayrgba.w) * (1.f - exp(-alpha));

    rayrgba += make_float4(rgb * contrib, alpha);
    t_acc += t * contrib;
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_mult_backward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    const float3& raysat,
    float fadescale,
    float fadeexp,
    tplate_t* grad_tplate,
    float* grad_warp,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale,
    float4& dL_rayrgba) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 xmt1 = raypos - w1t;
  float3 rxmt1 = w1r0 * xmt1.x + w1r1 * xmt1.y + w1r2 * xmt1.z;
  float3 y0 = rxmt1 * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));
    float3 dfade_y0 = fade * -(fadescale * fadeexp) *
        make_float3(pow(abs(y0.x), fadeexp - 1.f) * (y0.x > 0.f ? 1.f : -1.f),
                    pow(abs(y0.y), fadeexp - 1.f) * (y0.y > 0.f ? 1.f : -1.f),
                    pow(abs(y0.z), fadeexp - 1.f) * (y0.z > 0.f ? 1.f : -1.f));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    float alpha = sample.w * fade * stepsize;

    // move backward
    rayrgba.w = fmaxf(0.f, rayrgba.w - alpha);

    // accumulate
    float contrib = exp(-rayrgba.w) * (1.f - exp(-alpha));

    float3 dL_rgb = contrib * make_float3(dL_rayrgba);
    float dL_contrib = dot(rgb, make_float3(dL_rayrgba));

    float dL_alpha = exp(-rayrgba.w) * exp(-alpha) * dL_contrib + dL_rayrgba.w;
    float4 dL_sample = make_float4(dL_rgb, fade * stepsize * dL_alpha);

    dL_rayrgba.w = dL_rayrgba.w - contrib * dL_contrib;

    float3 dL_y0;
    if (do_warp) {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      float3 dL_y1 = GridSamplerBackwardT<float4, tplate_t>::exec(
          4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y1, dL_sample, false);

      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      float* grad_warp_ptr = grad_warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float3, float>::exec(
                  3, MD, MH, MW, warp_ptr, grad_warp_ptr, y0, dL_y1, false) +
          dfade_y0 * sample.w * stepsize * dL_alpha;
    } else {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float4, tplate_t>::exec(
                  4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y0, dL_sample, false) +
          dfade_y0 * sample.w * stepsize * dL_alpha;
    }

    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 0, rxmt1.x * dL_y0.x);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 1, rxmt1.y * dL_y0.y);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 2, rxmt1.z * dL_y0.z);

    dL_y0 *= w1s;

    float3 gw1r0 = xmt1.x * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 0, gw1r0.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 1, gw1r0.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 2, gw1r0.z);

    float3 gw1r1 = xmt1.y * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 0, gw1r1.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 1, gw1r1.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 2, gw1r1.z);

    float3 gw1r2 = xmt1.z * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 0, gw1r2.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 1, gw1r2.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 2, gw1r2.z);

    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 0, -dot(w1r0, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 1, -dot(w1r1, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 2, -dot(w1r2, dL_y0));
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_add2_forward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float t,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    float3* raysatim,
    float& t_acc,
    float fadescale,
    float fadeexp) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 y0 = raypos - w1t;
  y0 = (w1r0 * y0.x + w1r1 * y0.y + w1r2 * y0.z) * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    if (std::is_integral<tplate_t>::value) {
      rgb = tplate_ungamma(rgb);
    }
    float alpha = sample.w * fade * stepsize;

    // accumulate
    float contrib = fminf(1.f, rayrgba.w + alpha) - fminf(1.f, rayrgba.w);

    rayrgba += make_float4(rgb * contrib, alpha);
    t_acc += t * contrib;
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_add2_backward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    const float3& raysat,
    float fadescale,
    float fadeexp,
    tplate_t* grad_tplate,
    float* grad_warp,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale,
    float4& dL_rayrgba) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 xmt1 = raypos - w1t;
  float3 rxmt1 = w1r0 * xmt1.x + w1r1 * xmt1.y + w1r2 * xmt1.z;
  float3 y0 = rxmt1 * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));
    float3 dfade_y0 = fade * -(fadescale * fadeexp) *
        make_float3(pow(abs(y0.x), fadeexp - 1.f) * (y0.x > 0.f ? 1.f : -1.f),
                    pow(abs(y0.y), fadeexp - 1.f) * (y0.y > 0.f ? 1.f : -1.f),
                    pow(abs(y0.z), fadeexp - 1.f) * (y0.z > 0.f ? 1.f : -1.f));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    float alpha = sample.w * fade * stepsize;

    // move backward
    rayrgba.w -= alpha;

    // accumulate
    float newalpha = rayrgba.w + alpha;
    float contrib = fminf(1.f, rayrgba.w + alpha) - fminf(1.f, rayrgba.w);

    // move backward
    rayrgba -= make_float4(rgb * contrib, 0.f);

    float3 dL_rgb = contrib * make_float3(dL_rayrgba);
    float dL_contrib = dot(rgb, make_float3(dL_rayrgba));

    float dL_alpha = (rayrgba.w + alpha < 1.f ? 1.f : 0.f) * dL_contrib + dL_rayrgba.w;
    float4 dL_sample = make_float4(dL_rgb, fade * stepsize * dL_alpha);

    dL_rayrgba.w +=
        dL_contrib * ((rayrgba.w + alpha < 1.f ? 1.f : 0.f) - (rayrgba.w < 1.f ? 1.f : 0.f));

    float3 dL_y0;
    if (do_warp) {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      float3 dL_y1 = GridSamplerBackwardT<float4, tplate_t>::exec(
          4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y1, dL_sample, false);

      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      float* grad_warp_ptr = grad_warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float3, float>::exec(
                  3, MD, MH, MW, warp_ptr, grad_warp_ptr, y0, dL_y1, false) +
          dfade_y0 * sample.w * stepsize * dL_alpha;
    } else {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float4, tplate_t>::exec(
                  4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y0, dL_sample, false) +
          dfade_y0 * sample.w * stepsize * dL_alpha;
    }

    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 0, rxmt1.x * dL_y0.x);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 1, rxmt1.y * dL_y0.y);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 2, rxmt1.z * dL_y0.z);

    dL_y0 *= w1s;

    float3 gw1r0 = xmt1.x * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 0, gw1r0.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 1, gw1r0.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 2, gw1r0.z);

    float3 gw1r1 = xmt1.y * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 0, gw1r1.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 1, gw1r1.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 2, gw1r1.z);

    float3 gw1r2 = xmt1.z * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 0, gw1r2.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 1, gw1r2.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 2, gw1r2.z);

    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 0, -dot(w1r0, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 1, -dot(w1r1, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 2, -dot(w1r2, dL_y0));
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_mult2_forward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float t,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    float3* raysatim,
    float& t_acc,
    float fadescale,
    float fadeexp,
    float termthresh) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 y0 = raypos - w1t;
  y0 = (w1r0 * y0.x + w1r1 * y0.y + w1r2 * y0.z) * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    if (std::is_integral<tplate_t>::value) {
      rgb = tplate_ungamma(rgb);
    }
    float alpha = sample.w * fade * stepsize;

    // accumulate
    float rayrgbaw_next = rayrgba.w + alpha;
    // float fn0pre = (1.f - expf(-rayrgba.w / termthresh)) * termthresh;
    float fn0pre = expm1f(-rayrgba.w / termthresh) * -termthresh;
    float fn0 = fminf(1.f, fn0pre);
    // float fn1pre = (1.f - expf(-rayrgbaw_next / termthresh)) * termthresh;
    float fn1pre = expm1f(-rayrgbaw_next / termthresh) * -termthresh;
    float fn1 = fminf(1.f, fn1pre);
    float contrib = fn1 - fn0;

    rayrgba += make_float4(rgb * contrib, alpha);
    t_acc += t * contrib;

    if (fn1pre >= 1.f) {
      return 1;
    }
  }
  return 0;
}

template <
    bool do_warp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward,
    typename tplate_t>
static __forceinline__ __device__ bool prim_mult2_backward_eval(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    int n,
    int h,
    int w,
    int k,
    float stepsize,
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float4& rayrgba,
    const float3& raysat,
    float fadescale,
    float fadeexp,
    float termthresh,
    tplate_t* grad_tplate,
    float* grad_warp,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale,
    float4& dL_rayrgba) {
  float3 w1t = primpos[n * K + k];
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  float3 w1s = primscale[n * K + k];

  float3 xmt1 = raypos - w1t;
  float3 rxmt1 = w1r0 * xmt1.x + w1r1 * xmt1.y + w1r2 * xmt1.z;
  float3 y0 = rxmt1 * w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = exp(
        -fadescale * (pow(abs(y0.x), fadeexp) + pow(abs(y0.y), fadeexp) + pow(abs(y0.z), fadeexp)));
    float3 dfade_y0 = fade * -(fadescale * fadeexp) *
        make_float3(pow(abs(y0.x), fadeexp - 1.f) * (y0.x > 0.f ? 1.f : -1.f),
                    pow(abs(y0.y), fadeexp - 1.f) * (y0.y > 0.f ? 1.f : -1.f),
                    pow(abs(y0.z), fadeexp - 1.f) * (y0.z > 0.f ? 1.f : -1.f));

    float3 y1;
    if (do_warp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const tplate_t* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, tplate_t>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float3 rgb = make_float3(sample);
    float alpha = sample.w * fade * stepsize;

    // compute contrib
    float rayrgbaw_next = rayrgba.w;
    rayrgba.w = fmaxf(0.f, rayrgba.w - alpha);

    // float fn0pre = (1.f - expf(-rayrgba.w / termthresh)) * termthresh;
    float fn0pre = expm1f(-rayrgba.w / termthresh) * -termthresh;
    float fn0 = fminf(1.f, fn0pre);
    // float fn1pre = (1.f - expf(-rayrgbaw_next / termthresh)) * termthresh;
    float fn1pre = expm1f(-rayrgbaw_next / termthresh) * -termthresh;
    float fn1 = fminf(1.f, fn1pre);
    float contrib = fn1 - fn0;

    float dL_contrib = dot(rgb, make_float3(dL_rayrgba));

    float dL_fn0 = -dL_contrib;
    float dL_fn1 = dL_contrib;

    float dfn0_rayalpha = fn0pre < 1.f ? exp(-rayrgba.w / termthresh) : 0.f;
    float dfn1_alpha = fn1pre < 1.f ? exp(-rayrgbaw_next / termthresh) : 0.f;
    float dfn1_rayalpha = dfn1_alpha;
    float dL_alpha = dL_rayrgba.w + dfn1_alpha * dL_fn1;
    float3 dL_rgb = contrib * make_float3(dL_rayrgba);
    float4 dL_sample = make_float4(dL_rgb, fade * stepsize * dL_alpha);

    dL_rayrgba.w += dfn0_rayalpha * dL_fn0 + dfn1_rayalpha * dL_fn1;

    float3 dL_y0;
    if (do_warp) {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      float3 dL_y1 = GridSamplerBackwardT<float4, tplate_t>::exec(
          4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y1, dL_sample, false);

      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      float* grad_warp_ptr = grad_warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float3, float>::exec(
                  3, MD, MH, MW, warp_ptr, grad_warp_ptr, y0, dL_y1, false) +
          dfade_y0 * sample.w * stepsize * dL_alpha;
    } else {
      tplate_t* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float4, tplate_t>::exec(
                  4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y0, dL_sample, false) +
          dfade_y0 * sample.w * stepsize * dL_alpha;
    }

    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 0, rxmt1.x * dL_y0.x);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 1, rxmt1.y * dL_y0.y);
    atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 2, rxmt1.z * dL_y0.z);

    dL_y0 *= w1s;

    float3 gw1r0 = xmt1.x * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 0, gw1r0.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 1, gw1r0.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 2, gw1r0.z);

    float3 gw1r1 = xmt1.y * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 0, gw1r1.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 1, gw1r1.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 2, gw1r1.z);

    float3 gw1r2 = xmt1.z * dL_y0;

    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 0, gw1r2.x);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 1, gw1r2.y);
    atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 2, gw1r2.z);

    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 0, -dot(w1r0, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 1, -dot(w1r1, dL_y0));
    atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 2, -dot(w1r2, dL_y0));
  }
  return 0;
}
