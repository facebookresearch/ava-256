#pragma once

#include <array>

#define BOX_OVERLAP 512
#define N_WARPS 4

#include "mvpraymarch_kernel.h"

template <typename... T>
constexpr std::array<typename std::common_type<T...>::type, sizeof...(T)> make_array(T&&... t) {
  return {std::forward<T>(t)...};
}

constexpr int kernel_func_index(bool do_warp, int accum, bool chlast) {
  return 6 * chlast + 2 * accum + do_warp;
}

template <typename out, typename in>
using SampFwd = GridSamplerForward<out, in>;
template <typename out, typename in>
using SampBwd = GridSamplerBackward<out, in>;
template <typename out, typename in>
using SampFwdChl = GridSamplerChlastForward<out, in>;
template <typename out, typename in>
using SampBwdChl = GridSamplerChlastBackward<out, in>;

template <typename tplate_t>
constexpr auto get_fwd_kernel_func(bool do_warp, int accum, bool chlast) {
  static_assert(
      std::is_same<tplate_t, float>::value || std::is_same<tplate_t, uint8_t>::value,
      "Only float or uint8 templates are supported.");

  constexpr auto fwd_kernel_fn_ptrs = make_array(
      raymarch_fwd_kernel<false, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd>,
      raymarch_fwd_kernel<true, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd>,
      raymarch_fwd_kernel<false, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd>,
      raymarch_fwd_kernel<true, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd>,
      raymarch_fwd_kernel<false, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd>,
      raymarch_fwd_kernel<true, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd>,

      raymarch_fwd_kernel<false, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl>,
      raymarch_fwd_kernel<true, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl>,
      raymarch_fwd_kernel<false, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl>,
      raymarch_fwd_kernel<true, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl>,
      raymarch_fwd_kernel<false, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl>,
      raymarch_fwd_kernel<true, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl>);
  return fwd_kernel_fn_ptrs[kernel_func_index(do_warp, accum, chlast)];
}

template <typename tplate_t>
constexpr auto get_bwd_kernel_func(bool do_warp, int accum, bool chlast) {
  static_assert(
      std::is_same<tplate_t, float>::value || std::is_same<tplate_t, uint8_t>::value,
      "Only float or uint8 templates are supported.");

  constexpr auto bwd_kernel_fn_ptrs = make_array(
      raymarch_bwd_kernel<false, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd, SampBwd>,
      raymarch_bwd_kernel<true, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd, SampBwd>,
      raymarch_bwd_kernel<false, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd, SampBwd>,
      raymarch_bwd_kernel<true, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd, SampBwd>,
      raymarch_bwd_kernel<false, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd, SampBwd>,
      raymarch_bwd_kernel<true, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwd, SampBwd>,

      raymarch_bwd_kernel<false, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl, SampBwdChl>,
      raymarch_bwd_kernel<true, 0, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl, SampBwdChl>,
      raymarch_bwd_kernel<false, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl, SampBwdChl>,
      raymarch_bwd_kernel<true, 1, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl, SampBwdChl>,
      raymarch_bwd_kernel<false, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl, SampBwdChl>,
      raymarch_bwd_kernel<true, 2, BOX_OVERLAP, N_WARPS, tplate_t, SampFwdChl, SampBwdChl>);
  return bwd_kernel_fn_ptrs[kernel_func_index(do_warp, accum, chlast)];
}
