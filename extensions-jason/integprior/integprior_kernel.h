#include <cassert>

#include "../mvpraymarch/utils.h"

template <
    bool dowarp,
    int accum,
    template <typename, typename> class GridSamplerT = GridSamplerForward>
static __forceinline__ __device__ bool integprior_prim_forward_eval(
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
    const float* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float& rayalpha,
    float& rayprior,
    float3* raysatim,
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
    if (dowarp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const float* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, float>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float alpha = sample.w * fade * stepsize;

    // accumulate
    float rayrgbaw_next = rayalpha + alpha;
    rayalpha = rayrgbaw_next;

    float fn1, fn1pre;
    if (accum == 0) {
      fn1pre = rayrgbaw_next;
      fn1 = fminf(1.f, fn1pre);
    } else if (accum == 1) {
      fn1pre = (1.f - exp(-rayrgbaw_next));
      fn1 = fn1pre;
    } else if (accum == 2) {
      fn1pre = expm1f(-rayrgbaw_next / termthresh) * -termthresh;
      fn1 = fminf(1.f, fn1pre);
    }

    float p = (1.f - fn1) * fade;
    rayprior += p * stepsize;

    if (fn1pre >= 1.f) {
      return 1;
    }
  }
  return 0;
}

template <
    bool dowarp,
    int accum,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward>
static __forceinline__ __device__ bool integprior_prim_backward_eval(
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
    const float* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float3& raypos,
    float& rayalpha,
    const float3& raysat,
    float fadescale,
    float fadeexp,
    float termthresh,
    float* grad_tplate,
    float* grad_warp,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale,
    float& dL_rayalpha,
    float dL_p) {
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
    if (dowarp) {
      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      y1 = GridSamplerT<float3, float>::exec(3, MD, MH, MW, warp_ptr, y0, false);
    } else {
      y1 = y0;
    }

    const float* tplate_ptr = tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
    float4 sample = GridSamplerT<float4, float>::exec(4, MD, MH, MW, tplate_ptr, y1, false);

    float alpha = sample.w * fade * stepsize;

    // compute contrib
    float rayrgbaw_next = rayalpha;
    rayalpha = fmaxf(0.f, rayalpha - alpha);

    float dL_fn1 = -fade * dL_p;

    float fn1;
    if (accum == 0) {
      float fn1pre = rayrgbaw_next;
      fn1 = fminf(1.f, fn1pre);
      dL_rayalpha += fn1pre < 1.f ? dL_fn1 : 0.f;
    } else if (accum == 1) {
      fn1 = (1.f - exp(-rayrgbaw_next));
      dL_rayalpha += exp(-rayrgbaw_next) * dL_fn1;
    } else if (accum == 2) {
      float fn1pre = expm1f(-rayrgbaw_next / termthresh) * -termthresh;
      fn1 = fminf(1.f, fn1pre);
      dL_rayalpha += fn1pre < 1.f ? exp(-rayrgbaw_next / termthresh) * dL_fn1 : 0.f;
    }

    float dL_alpha = dL_rayalpha;

    float dL_fade = (1.f - fn1) * dL_p + sample.w * stepsize * dL_alpha;

    float4 dL_sample = make_float4(make_float3(0.f), fade * stepsize * dL_alpha);

    float3 dL_y0;
    if (dowarp) {
      float* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      float3 dL_y1 = GridSamplerBackwardT<float4, float>::exec(
          4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y1, dL_sample, false);

      const float* warp_ptr = warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      float* grad_warp_ptr = grad_warp + (n * K * 3 * MD * MH * MW) + (k * 3 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float3, float>::exec(
                  3, MD, MH, MW, warp_ptr, grad_warp_ptr, y0, dL_y1, false) +
          dfade_y0 * dL_fade;
    } else {
      float* grad_tplate_ptr = grad_tplate + (n * K * 4 * MD * MH * MW) + (k * 4 * MD * MH * MW);
      dL_y0 = GridSamplerBackwardT<float4, float>::exec(
                  4, MD, MH, MW, tplate_ptr, grad_tplate_ptr, y0, dL_sample, false) +
          dfade_y0 * dL_fade;
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
    bool dowarp,
    int accum,
    int maxhitboxes,
    template <int> class RayBVHSubsetT = RayBVHSubsetSync,
    template <typename, typename> class GridSamplerT = GridSamplerForward>
__global__ void integprior_fwd_kernel(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    const float3* rayposim,
    const float3* raydirim,
    float stepsize,
    const float2* tminmaxim,
    const int* sortedobjid,
    const int2* nodechildren,
    const float3* nodeaabb,
    const float* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    float* rayrgbaim,
    float* priorim,
    float3* raysatim,
    float fadescale,
    float fadeexp,
    float termthresh) {
  bool validthread = false;
  int w, h, n;
  if (gridDim.z != 1) {
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = blockIdx.y * blockDim.y + threadIdx.y;
    n = blockIdx.z * blockDim.z + threadIdx.z;
    validthread = (w < W) && (h < H) && (n < N);
  } else if (gridDim.y != 1) {
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = (blockIdx.y * blockDim.y + threadIdx.y) % H;
    n = (blockIdx.y * blockDim.y + threadIdx.y) / H;
    validthread = (w < W) && (h < H) && (n < N);
  } else {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    w = index % W;
    h = (index / W) % H;
    n = index / (H * W);
    validthread = index < N * H * W;
  }

  // warpmask contains the valid threads in the warp
  unsigned warpmask = __ballot_sync(0xffffffff, validthread);

  if (validthread) {
    float3 raypos = rayposim[n * H * W + h * W + w];
    const float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];

    float rayalpha = 0.f;
    float rayprior = 0.f;
    int hitboxes[maxhitboxes];
    int nhitboxes = 0;

    // find raytminmax
    float2 rtminmax = make_float2(max(tminmax.x, tminmax.y), min(tminmax.x, tminmax.y));
    RayBVHSubsetT<maxhitboxes>::exec(
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
        nhitboxes);

    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;
    if (nhitboxes) {
      int incs = int(floor((rtminmax.x - t) / stepsize));

      t += incs * stepsize;
      raypos += raydir * incs * stepsize;

      int nsteps = 0;

      bool done = false;
      while (t < rtminmax.y + stepsize) {
        for (int ks = 0; ks < nhitboxes; ++ks) {
          int k = hitboxes[ks];

          if (integprior_prim_forward_eval<dowarp, accum, GridSamplerT>(
                  N,
                  H,
                  W,
                  K,
                  MD,
                  MH,
                  MW,
                  n,
                  h,
                  w,
                  k,
                  stepsize,
                  tplate,
                  warp,
                  primpos,
                  primrot,
                  primscale,
                  raypos,
                  rayalpha,
                  rayprior,
                  raysatim,
                  fadescale,
                  fadeexp,
                  termthresh)) {
            if (accum == 0) {
              done = true;
            } else if (accum == 1) {
              done =
                  termthresh > 0.f && rayalpha > -logf(termthresh) || 1.f - exp(-rayalpha) == 1.f;
            } else if (accum == 2) {
              done = true;
            }
          }
        }

        // update position
        ++nsteps;
        t += stepsize;
        raypos += raydir * stepsize;

        if (done) {
          break;
        }
      }

      raysatim[n * H * W + h * W + w] = make_float3((float)nsteps, 0.f, 0.f);
    }

    rayrgbaim[n * H * W + h * W + w] = rayalpha;
    priorim[n * H * W + h * W + w] = rayprior;
  }
}

template <
    bool dowarp,
    int accum,
    int maxhitboxes,
    template <int> class RayBVHSubsetT = RayBVHSubsetSync,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward>
__global__ void integprior_bwd_kernel(
    int N,
    int H,
    int W,
    int K,
    int MD,
    int MH,
    int MW,
    const float3* rayposim,
    const float3* raydirim,
    float stepsize,
    const float2* tminmaxim,
    const int* sortedobjid,
    const int2* nodechildren,
    const float3* nodeaabb,
    const float* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float* rayrgbaim,
    const float3* raysatim,
    const float* grad_prior,
    float* grad_tplate,
    float* grad_warp,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale,
    float fadescale,
    float fadeexp,
    float termthresh) {
  bool validthread = false;
  int w, h, n;
  if (gridDim.z != 1) {
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = blockIdx.y * blockDim.y + threadIdx.y;
    n = blockIdx.z * blockDim.z + threadIdx.z;
    validthread = (w < W) && (h < H) && (n < N);
  } else if (gridDim.y != 1) {
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = (blockIdx.y * blockDim.y + threadIdx.y) % H;
    n = (blockIdx.y * blockDim.y + threadIdx.y) / H;
    validthread = (w < W) && (h < H) && (n < N);
  } else {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    w = index % W;
    h = (index / W) % H;
    n = index / (H * W);
    validthread = index < N * H * W;
  }

  // warpmask contains the valid threads in the warp
  unsigned warpmask = __ballot_sync(0xffffffff, validthread);

  if (validthread) {
    float3 raypos = rayposim[n * H * W + h * W + w];
    float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];

    float rayalpha = rayrgbaim[n * H * W + h * W + w];
    float dL_rayalpha = 0.f;
    float dL_p = grad_prior[n * H * W + h * W + w] * stepsize;
    float3 raysat = raysatim[n * H * W + h * W + w];

    int hitboxes[maxhitboxes];
    int nhitboxes = 0;

    // find raytminmax
    float2 rtminmax = make_float2(max(tminmax.x, tminmax.y), min(tminmax.x, tminmax.y));
    RayBVHSubsetT<maxhitboxes>::exec(
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
        nhitboxes);

    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;

    if (nhitboxes == 0) {
      return;
    }

    int incs = int(floor((rtminmax.x - t) / stepsize));

    t += incs * stepsize;
    raypos += raydir * incs * stepsize;

    int nsteps = (int)raysatim[n * H * W + h * W + w].x;
    t += nsteps * stepsize;
    raypos += raydir * nsteps * stepsize;

    while (--nsteps >= 0) {
      // update position
      t += -stepsize;
      raypos += raydir * -stepsize;

      for (int ks = 0; ks < nhitboxes; ++ks) {
        int k = hitboxes[nhitboxes - ks - 1];
        if (integprior_prim_backward_eval<dowarp, accum, GridSamplerT, GridSamplerBackwardT>(
                N,
                H,
                W,
                K,
                MD,
                MH,
                MW,
                n,
                h,
                w,
                k,
                stepsize,
                tplate,
                warp,
                primpos,
                primrot,
                primscale,
                raypos,
                rayalpha,
                raysat,
                fadescale,
                fadeexp,
                termthresh,
                grad_tplate,
                grad_warp,
                grad_primpos,
                grad_primrot,
                grad_primscale,
                dL_rayalpha,
                dL_p)) {
        }
      }
    }
  }
}
