#pragma once

template <
    bool dowarp,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    typename tplate_t>
static __forceinline__ __device__ bool fast_prim_forward_eval(
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
  float3 rmt = raypos - w1t;
  float3 w1r0 = primrot[(n * K + k) * 3 + 0];
  float3 y0 = w1r0 * rmt.x;
  float3 w1r1 = primrot[(n * K + k) * 3 + 1];
  y0 += w1r1 * rmt.y;
  float3 w1r2 = primrot[(n * K + k) * 3 + 2];
  y0 += w1r2 * rmt.z;
  float3 w1s = primscale[n * K + k];
  y0 *= w1s;

  bool validk = y0.x > -1.f && y0.x < 1.f && y0.y > -1.f && y0.y < 1.f && y0.z > -1.f && y0.z < 1.f;

  if (validk) {
    float fade = __expf(
        -fadescale *
        (__powf(abs(y0.x), fadeexp) + __powf(abs(y0.y), fadeexp) + __powf(abs(y0.z), fadeexp)));

    float3 y1;
    if (dowarp) {
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

template <int maxhitboxes>
static __forceinline__ __device__ void ray_bvh_tminmax_subset_fast_sync(
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
    // check if we're in a leaf
    if (node >= (K - 1)) {
      // for (int ks = k0; ks < k1; ++ks) {
      {
        int k = node - (K - 1);

        float3 w1t = primpos[n * K + k];
        float3 rmt = raypos - w1t;
        float3 w1r0 = primrot[(n * K + k) * 3 + 0];
        float3 r0 = w1r0 * rmt.x;
        float3 rd = w1r0 * raydir.x;
        float3 w1r1 = primrot[(n * K + k) * 3 + 1];
        r0 += w1r1 * rmt.y;
        rd += w1r1 * raydir.y;
        float3 w1r2 = primrot[(n * K + k) * 3 + 2];
        r0 += w1r2 * rmt.z;
        rd += w1r2 * raydir.z;
        float3 w1s = primscale[n * K + k];

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

        bool intersection = trmin <= trmax;

        if (intersection) {
          // hit
          rtminmax.x = fminf(rtminmax.x, trmin);
          rtminmax.y = fmaxf(rtminmax.y, trmax);
        }

        intersection = __any_sync(warpmask, intersection);

        if (intersection) {
          if (num < maxhitboxes) {
            hitboxes[num] = k; // all threads will write simultaneously but that's okay
            ++num;
          }
        }
      }

      node = *--stack_ptr;
    } else {
      int2 children = make_int2(node * 2 + 1, node * 2 + 2);

      // check if we're in each child's bbox
      const float3* nodeaabb_ptr = nodeaabb + n * (K + K - 1) * 2 + children.x * 2;
      bool traverse_l = ray_aabb_hit_ird(nodeaabb_ptr[0], nodeaabb_ptr[1], raypos, iraydir);
      bool traverse_r = ray_aabb_hit_ird(nodeaabb_ptr[2], nodeaabb_ptr[3], raypos, iraydir);

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

template <
    bool dowarp,
    int accum,
    int maxhitboxes,
    int nwarps,
    typename tplate_t,
    template <typename, typename> class GridSamplerT = GridSamplerForward>
__global__ void raymarch_fwd_kernel(
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
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    typename vec_t<tplate_t, 4>::type* rayrgbaim,
    float3* raysatim,
    int2* raytermim,
    float* t_im,
    bool broadcast_slab,
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

  int warpid = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

  // warpmask contains the valid threads in the warp
  unsigned warpmask = __ballot_sync(0xffffffff, validthread);

  const int n_bcast = broadcast_slab ? 0 : n;
  auto raysatim_ptr = raysatim ? (raysatim + n * H * W + h * W + w) : nullptr;

  if (validthread) {
    float3 raypos = rayposim[n * H * W + h * W + w];
    const float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];
    float t_accum = 0.f;

    float4 rayrgba = make_float4(0.f);
    __shared__ int hitboxes[maxhitboxes * nwarps];
    int* hitboxes_ptr = hitboxes + maxhitboxes * warpid;
    int nhitboxes = 0;

    // find raytminmax
    float2 rtminmax = make_float2(max(tminmax.x, tminmax.y), min(tminmax.x, tminmax.y));
    ray_bvh_tminmax_subset_fast_sync<maxhitboxes>(
        warpmask,
        K,
        n_bcast,
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
        hitboxes_ptr,
        nhitboxes);
    __syncwarp(warpmask);

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
          int k = hitboxes_ptr[ks];

          if (accum == 0) {
            if (fast_prim_forward_eval<dowarp, GridSamplerT>(
                    N,
                    H,
                    W,
                    K,
                    MD,
                    MH,
                    MW,
                    n_bcast,
                    h,
                    w,
                    k,
                    t,
                    stepsize,
                    tplate,
                    warp,
                    primpos,
                    primrot,
                    primscale,
                    raypos,
                    rayrgba,
                    raysatim_ptr,
                    t_accum,
                    fadescale,
                    fadeexp)) {
              // t = tminmax.y;
              done = true;
              break; // break mid loop, it don't matter
            }
          } else if (accum == 1) {
            if (prim_mult_forward_eval<dowarp, GridSamplerT>(
                    N,
                    H,
                    W,
                    K,
                    MD,
                    MH,
                    MW,
                    n_bcast,
                    h,
                    w,
                    k,
                    t,
                    stepsize,
                    tplate,
                    warp,
                    primpos,
                    primrot,
                    primscale,
                    raypos,
                    rayrgba,
                    nullptr,
                    t_accum,
                    fadescale,
                    fadeexp)) {
            }
            done =
                termthresh > 0.f && rayrgba.w > -logf(termthresh) || 1.f - exp(-rayrgba.w) == 1.f;
          } else if (accum == 2) {
            if (prim_mult2_forward_eval<dowarp, GridSamplerT>(
                    N,
                    H,
                    W,
                    K,
                    MD,
                    MH,
                    MW,
                    n_bcast,
                    h,
                    w,
                    k,
                    t,
                    stepsize,
                    tplate,
                    warp,
                    primpos,
                    primrot,
                    primscale,
                    raypos,
                    rayrgba,
                    nullptr,
                    t_accum,
                    fadescale,
                    fadeexp,
                    termthresh)) {
              done = true;
              // break; // do not break mid loop, don't do it.
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

      if (accum > 0 && raytermim) {
        raytermim[n * H * W + h * W + w] = make_int2(nsteps, 0);
      }
    }

    // NOTE: None of this works w/the backward pass, but then again we
    // can't really do autograd w/integral types so it doesn't matter.
    float3 final_rgb = make_float3(rayrgba.x, rayrgba.y, rayrgba.z);
    float final_alpha = rayrgba.w;
    if (std::is_integral<tplate_t>::value) {
      if (accum == 1) {
        final_alpha = 1.f - exp(-final_alpha);
      } else if (accum == 2) {
        final_alpha = termthresh * (1.f - exp(-final_alpha / termthresh));
      }
      final_alpha = min(max(final_alpha, 0.f), 1.f);
      final_alpha *= (float)std::numeric_limits<tplate_t>::max();
      final_rgb = output_color_curve(final_rgb);
    }

    rayrgbaim[n * H * W + h * W + w] = {
        (tplate_t)final_rgb.x,
        (tplate_t)final_rgb.y,
        (tplate_t)final_rgb.z,
        (tplate_t)final_alpha,
    };

    if (t_im) {
      t_im[n * H * W + h * W + w] = t_accum;
    }
  }
}

template <
    bool dowarp,
    int accum,
    int maxhitboxes,
    int nwarps,
    typename tplate_t,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward>
__global__ void raymarch_bwd_kernel(
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
    const tplate_t* tplate,
    const float* warp,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const typename vec_t<tplate_t, 4>::type* rayrgbaim,
    const float3* raysatim,
    const int2* raytermim,
    const float4* grad_rayrgba,
    tplate_t* grad_tplate,
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

  int warpid = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

  // warpmask contains the valid threads in the warp
  unsigned warpmask = __ballot_sync(0xffffffff, validthread);

  if (validthread) {
    float3 raypos = rayposim[n * H * W + h * W + w];
    float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];

    float4 rayrgba = accum > 0 ? rayrgbaim[n * H * W + h * W + w] : make_float4(0.f);
    float4 dL_rayrgba = grad_rayrgba[n * H * W + h * W + w];
    float3 raysat = accum == 0 ? raysatim[n * H * W + h * W + w] : make_float3(0.f);
    int2 rayterm = accum > 0 ? raytermim[n * H * W + h * W + w] : make_int2(0);

    __shared__ int hitboxes[maxhitboxes * nwarps];
    int* hitboxes_ptr = hitboxes + maxhitboxes * warpid;
    int nhitboxes = 0;

    // find raytminmax
    float2 rtminmax = make_float2(max(tminmax.x, tminmax.y), min(tminmax.x, tminmax.y));
    ray_bvh_tminmax_subset_fast_sync<maxhitboxes>(
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
        hitboxes_ptr,
        nhitboxes);
    __syncwarp(warpmask);

    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;

    if (nhitboxes == 0) {
      return;
    }

    int incs = int(floor((rtminmax.x - t) / stepsize));

    t += incs * stepsize;
    raypos += raydir * incs * stepsize;

    int nsteps = 0;
    if (accum > 0) {
      nsteps = rayterm.x;
      t += nsteps * stepsize;
      raypos += raydir * nsteps * stepsize;
    }

    while (accum > 0 ? --nsteps >= 0 : t < rtminmax.y + stepsize) {
      if (accum > 0) {
        // update position
        t += -stepsize;
        raypos += raydir * -stepsize;
      }

      for (int ks = 0; ks < nhitboxes; ++ks) {
        int k = hitboxes_ptr[accum > 0 ? nhitboxes - ks - 1 : ks];
        if (accum == 0) {
          if (prim_backward_eval<dowarp, GridSamplerT, GridSamplerBackwardT>(
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
                  rayrgba,
                  raysat,
                  fadescale,
                  fadeexp,
                  grad_tplate,
                  grad_warp,
                  grad_primpos,
                  grad_primrot,
                  grad_primscale,
                  dL_rayrgba)) {
            t = tminmax.y;
            break;
          }
        } else if (accum == 1) {
          if (prim_mult_backward_eval<dowarp, GridSamplerT, GridSamplerBackwardT>(
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
                  rayrgba,
                  raysat,
                  fadescale,
                  fadeexp,
                  grad_tplate,
                  grad_warp,
                  grad_primpos,
                  grad_primrot,
                  grad_primscale,
                  dL_rayrgba)) {
          }
        } else if (accum == 2) {
          if (prim_mult2_backward_eval<dowarp, GridSamplerT, GridSamplerBackwardT>(
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
                  rayrgba,
                  raysat,
                  fadescale,
                  fadeexp,
                  termthresh,
                  grad_tplate,
                  grad_warp,
                  grad_primpos,
                  grad_primrot,
                  grad_primscale,
                  dL_rayrgba)) {
          }
        }
      }

      if (accum == 0) {
        // update position
        t += stepsize;
        raypos += raydir * stepsize;
      }
    }
  }
}
