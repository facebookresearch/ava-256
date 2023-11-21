template <
    bool dowarp,
    int accum,
    typename tplate_t,
    template <typename, typename> class GridSamplerT = GridSamplerForward>
__global__ void raymarch_nobvh_forward_kernel(
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

  const int n_bcast = broadcast_slab ? 0 : n;
  auto raysatim_ptr = raysatim ? (raysatim + n * H * W + h * W + w) : nullptr;

  if (validthread) {
    float3 raypos = rayposim[n * H * W + h * W + w];
    const float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];
    float t_accum = 0.f;

    float4 rayrgba = make_float4(0.f);

    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;

    int nsteps = 0;

    bool done = false;
    while (t < tminmax.y) {
      for (int k = 0; k < K; ++k) {
        if (accum == 0) {
          if (prim_forward_eval<dowarp, GridSamplerT>(
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
            break;
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
          done = termthresh > 0.f && rayrgba.w > -logf(termthresh);
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
            // break;
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
    typename tplate_t,
    template <typename, typename> class GridSamplerT = GridSamplerForward,
    template <typename, typename> class GridSamplerBackwardT = GridSamplerBackward>
__global__ void raymarch_nobvh_backward_kernel(
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

  if (validthread) {
    float3 raypos = rayposim[n * H * W + h * W + w];
    float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];

    float4 rayrgba = accum > 0 ? rayrgbaim[n * H * W + h * W + w] : make_float4(0.f);
    float4 dL_rayrgba = grad_rayrgba[n * H * W + h * W + w];
    float3 raysat = accum == 0 ? raysatim[n * H * W + h * W + w] : make_float3(0.f);
    int2 rayterm = accum > 0 ? raytermim[n * H * W + h * W + w] : make_int2(0);

    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;

    int nsteps = 0;
    if (accum > 0) {
      nsteps = rayterm.x;
      t += nsteps * stepsize;
      raypos += raydir * nsteps * stepsize;
    }

    while (accum > 0 ? --nsteps >= 0 : t < tminmax.y) {
      if (accum > 0) {
        // update position
        t -= stepsize;
        raypos -= raydir * stepsize;
      }

      for (int k = 0; k < K; ++k) {
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
                  K - k - 1,
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
                  K - k - 1,
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
