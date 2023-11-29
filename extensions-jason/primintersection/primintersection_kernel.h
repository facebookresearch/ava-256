#include <cassert>

#include "../mvpraymarch/utils.h"

__global__ void primintersection_forward_kernel(
    int N,
    int H,
    int W,
    int K,
    const float3* rayposim,
    const float3* raydirim,
    const int* sortedobjid,
    const int2* nodechildren,
    const float3* nodeaabb,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    float* raystepsim) {
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
    const float3 raydir = raydirim[n * H * W + h * W + w];

    float raysteps = 0.f;
    float3 iraydir = 1.0f / raydir;
    int stack[64];
    int* stack_ptr = stack;
    *stack_ptr++ = -1;
    int node = 0;
    do {
      int2 children = nodechildren[n * (K + K - 1) + node];

      // check if we're in a leaf
      if (children.x < 0) {
        int k0 = -children.x - 1, k1 = -children.y - 1;

        for (int ks = k0; ks < k1; ++ks) {
          int k = sortedobjid[n * K + ks];

          float3 w1t = primpos[n * K + k];
          float3 w1r0 = primrot[(n * K + k) * 3 + 0];
          float3 w1r1 = primrot[(n * K + k) * 3 + 1];
          float3 w1r2 = primrot[(n * K + k) * 3 + 2];
          float3 w1s = primscale[n * K + k];

          float3 y0 = raypos - w1t;
          y0 = (w1r0 * y0.x + w1r1 * y0.y + w1r2 * y0.z) * w1s;

          float3 yd = raydir;
          yd = (w1r0 * yd.x + w1r1 * yd.y + w1r2 * yd.z) * w1s;

          float a = dot(yd, yd);
          float b = -2.f * dot(yd, y0);
          float c = dot(y0, y0);

          float sqrta = sqrt(a);
          float bsq = b * b;

          float kk = c - 0.25f * bsq / a;
          const float r = 4.f;
          float disc = max(1e-8f, bsq - 4.f * a * (c - r));
          float e0 = erff(-0.5f * sqrt(disc / a));
          float e1 = erff(0.5f * sqrt(disc / a));
          const float sqrtpi = 1.7724538f;
          float v = sqrtpi * 0.5f * exp(-kk) * (e1 - e0) / sqrta;

          raysteps += v;
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
            if (stack_ptr - stack >= 64) {
              printf("stack_ptr exceeded!\n");
            }
            assert(stack_ptr - stack < 64);
          }
        }
      }
    } while (node != -1);

    raystepsim[n * H * W + h * W + w] = raysteps;
  }
}

__global__ void primintersection_backward_kernel(
    int N,
    int H,
    int W,
    int K,
    const float3* rayposim,
    const float3* raydirim,
    const int* sortedobjid,
    const int2* nodechildren,
    const float3* nodeaabb,
    const float3* primpos,
    const float3* primrot,
    const float3* primscale,
    const float* raystepsim,
    const float* grad_raysteps,
    float3* grad_primpos,
    float3* grad_primrot,
    float3* grad_primscale) {
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

    float dL_raysteps = grad_raysteps[n * H * W + h * W + w];

    float3 iraydir = 1.0f / raydir;
    int stack[64];
    int* stack_ptr = stack;
    *stack_ptr++ = -1;
    int node = 0;
    do {
      int2 children = nodechildren[n * (K + K - 1) + node];

      // check if we're in a leaf
      if (children.x < 0) {
        int k0 = -children.x - 1, k1 = -children.y - 1;

        for (int ks = k0; ks < k1; ++ks) {
          int k = sortedobjid[n * K + ks];

          float3 w1t = primpos[n * K + k];
          float3 w1r0 = primrot[(n * K + k) * 3 + 0];
          float3 w1r1 = primrot[(n * K + k) * 3 + 1];
          float3 w1r2 = primrot[(n * K + k) * 3 + 2];
          float3 w1s = primscale[n * K + k];

          float3 xmt1 = raypos - w1t;
          float3 rxmt1 = w1r0 * xmt1.x + w1r1 * xmt1.y + w1r2 * xmt1.z;
          float3 y0 = rxmt1 * w1s;

          float3 xmt1d = raydir;
          float3 rxmt1d = w1r0 * xmt1d.x + w1r1 * xmt1d.y + w1r2 * xmt1d.z;
          float3 yd = rxmt1d * w1s;

          float a = dot(yd, yd);
          float b = -2.f * dot(yd, y0);
          float c = dot(y0, y0);

          float sqrta = sqrt(a);
          float bsq = b * b;

          float kk = c - 0.25f * bsq / a;
          // float h = -0.5f * b / a;
          const float r = 4.f;
          // float disc = max(0.f, bsq - 4.f * a * (c - r));
          float disc_ = max(1e-8f, bsq / a - 4.f * (c - r));
          // float t0 = 0.5f * (-b - sqrt(disc)) / a;
          // float t1 = 0.5f * (-b + sqrt(disc)) / a;
          // float t0 = h - sqrt(disc) / a;
          // float t1 = h + sqrt(disc) / a;
          // float e0 = erff(sqrt(a) * (t0 - h));
          // float e1 = erff(sqrt(a) * (t1 - h));
          // float e0 = erff(-0.5f * sqrt(disc / a));
          // float e1 = erff( 0.5f * sqrt(disc / a));
          float e0 = erff(-0.5f * sqrt(disc_));
          float e1 = erff(0.5f * sqrt(disc_));
          const float sqrtpi = 1.7724538f;
          float v = sqrtpi * 0.5f * exp(-kk) * (e1 - e0) / sqrta;

          //// dL_v
          // float dL_a = sqrtpi * 0.5f * exp(-kk) * (e1 - e0) * (-0.5f * pow(a, -1.5f)) *
          // dL_raysteps; float dL_kk = sqrtpi * 0.5f * exp(-kk) * (e1 - e0) / sqrta * (-1.f) *
          // dL_raysteps; float dL_e0 = sqrtpi * 0.5f * exp(-kk) / sqrta * (-1.f) * dL_raysteps;
          // float dL_e1 = sqrtpi * 0.5f * exp(-kk) / sqrta * ( 1.f) * dL_raysteps;

          //// dL_e0
          // dL_a += 2.f / sqrtpi * exp(-0.25f * disc / a) * (-0.5f * sqrt(disc) * -0.5f * pow(a,
          // -1.5f)) * dL_e0; float dL_disc = 2.f / sqrtpi * exp(-0.25f * disc / a) * (-0.5f * 0.5f *
          // pow(disc, -0.5f) / sqrta) * dL_e0;

          //// dL_e1
          // dL_a += 2.f / sqrtpi * exp(-0.25f * disc / a) * (0.5f * sqrt(disc) * -0.5f * pow(a,
          // -1.5f)) * dL_e1; dL_disc += 2.f / sqrtpi * exp(-0.25f * disc / a) * (0.5f * 0.5f *
          // pow(disc, -0.5f) / sqrta) * dL_e1;

          //// dL_kk
          // dL_a += -(c - 0.25f * b * b) / (a * a) * dL_kk;
          // float dL_b = -0.25f / a * (2.f * b) * dL_kk;
          // float dL_c = 1.f / a * dL_kk;

          //// dL_disc
          // dL_a += (disc > 0.f ? -4.f * (c - r) : 0.f) * dL_disc;
          // dL_b += (disc > 0.f ? 2.f * b : 0.f) * dL_disc;
          // dL_c += (disc > 0.f ? -4.f * a : 0.f) * dL_disc;

          // dL_v
          float dL_a = sqrtpi * 0.5f * exp(-kk) * (e1 - e0) * (-0.5f * pow(a, -1.5f)) * dL_raysteps;
          float dL_kk = sqrtpi * 0.5f * exp(-kk) * (e1 - e0) / sqrta * (-1.f) * dL_raysteps;
          float dL_e0 = sqrtpi * 0.5f * exp(-kk) / sqrta * (-1.f) * dL_raysteps;
          float dL_e1 = sqrtpi * 0.5f * exp(-kk) / sqrta * (1.f) * dL_raysteps;

          // dL_e0
          float dL_disc =
              2.f / sqrtpi * exp(-0.25f * disc_) * (-0.5f * 0.5f * pow(disc_, -0.5f)) * dL_e0;

          // dL_e1
          dL_disc += 2.f / sqrtpi * exp(-0.25f * disc_) * (0.5f * 0.5f * pow(disc_, -0.5f)) * dL_e1;

          // dL_kk
          dL_a += 0.25f * bsq / (a * a) * dL_kk;
          float dL_b = -0.25f / a * (2.f * b) * dL_kk;
          float dL_c = 1.f * dL_kk;

          // dL_disc
          dL_a += (disc_ > 1e-8f ? -b * b / (a * a) : 0.f) * dL_disc;
          dL_b += (disc_ > 1e-8f ? 2.f * b / a : 0.f) * dL_disc;
          dL_c += (disc_ > 1e-8f ? -4.f : 0.f) * dL_disc;

          float3 dL_y0 = -2.f * yd * dL_b + 2.f * y0 * dL_c;
          float3 dL_yd = -2.f * y0 * dL_b + 2.f * yd * dL_a;

          float3 dL_s = rxmt1 * dL_y0 + rxmt1d * dL_yd;
          atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 0, dL_s.x);
          atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 1, dL_s.y);
          atomicAdd((float*)grad_primscale + (n * K + k) * 3 + 2, dL_s.z);

          dL_y0 *= w1s;
          dL_yd *= w1s;

          float3 gw1r0 = xmt1.x * dL_y0 + xmt1d.x * dL_yd;

          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 0, gw1r0.x);
          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 1, gw1r0.y);
          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 0) * 3 + 2, gw1r0.z);

          float3 gw1r1 = xmt1.y * dL_y0 + xmt1d.y * dL_yd;

          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 0, gw1r1.x);
          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 1, gw1r1.y);
          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 1) * 3 + 2, gw1r1.z);

          float3 gw1r2 = xmt1.z * dL_y0 + xmt1d.z * dL_yd;

          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 0, gw1r2.x);
          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 1, gw1r2.y);
          atomicAdd((float*)grad_primrot + ((n * K + k) * 3 + 2) * 3 + 2, gw1r2.z);

          atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 0, -dot(w1r0, dL_y0));
          atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 1, -dot(w1r1, dL_y0));
          atomicAdd((float*)grad_primpos + (n * K + k) * 3 + 2, -dot(w1r2, dL_y0));
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
            if (stack_ptr - stack >= 64) {
              printf("stack_ptr exceeded!\n");
            }
            assert(stack_ptr - stack < 64);
          }
        }
      }
    } while (node != -1);
  }
}
