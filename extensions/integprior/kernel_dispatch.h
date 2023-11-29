#include <array>

#define BOX_OVERLAP 512
#define N_WARPS 4

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

template <int maxhitboxes>
using Subset = RayBVHSubsetSync<maxhitboxes>;

constexpr auto get_integprior_fwd_kernel_func(bool do_warp, int accum, bool chlast) {
  constexpr auto integprior_fwd_kernel_fn_ptrs = make_array(
      integprior_fwd_kernel<false, 0, BOX_OVERLAP, Subset, SampFwd>,
      integprior_fwd_kernel<true, 0, BOX_OVERLAP, Subset, SampFwd>,
      integprior_fwd_kernel<false, 1, BOX_OVERLAP, Subset, SampFwd>,
      integprior_fwd_kernel<true, 1, BOX_OVERLAP, Subset, SampFwd>,
      integprior_fwd_kernel<false, 2, BOX_OVERLAP, Subset, SampFwd>,
      integprior_fwd_kernel<true, 2, BOX_OVERLAP, Subset, SampFwd>,

      integprior_fwd_kernel<false, 0, BOX_OVERLAP, Subset, SampFwdChl>,
      integprior_fwd_kernel<true, 0, BOX_OVERLAP, Subset, SampFwdChl>,
      integprior_fwd_kernel<false, 1, BOX_OVERLAP, Subset, SampFwdChl>,
      integprior_fwd_kernel<true, 1, BOX_OVERLAP, Subset, SampFwdChl>,
      integprior_fwd_kernel<false, 2, BOX_OVERLAP, Subset, SampFwdChl>,
      integprior_fwd_kernel<true, 2, BOX_OVERLAP, Subset, SampFwdChl>);
  return integprior_fwd_kernel_fn_ptrs[kernel_func_index(do_warp, accum, chlast)];
}

constexpr auto get_integprior_bwd_kernel_func(bool do_warp, int accum, bool chlast) {
  constexpr auto integprior_bwd_kernel_fn_ptrs = make_array(
      integprior_bwd_kernel<false, 0, BOX_OVERLAP, Subset, SampFwd, SampBwd>,
      integprior_bwd_kernel<true, 0, BOX_OVERLAP, Subset, SampFwd, SampBwd>,
      integprior_bwd_kernel<false, 1, BOX_OVERLAP, Subset, SampFwd, SampBwd>,
      integprior_bwd_kernel<true, 1, BOX_OVERLAP, Subset, SampFwd, SampBwd>,
      integprior_bwd_kernel<false, 2, BOX_OVERLAP, Subset, SampFwd, SampBwd>,
      integprior_bwd_kernel<true, 2, BOX_OVERLAP, Subset, SampFwd, SampBwd>,

      integprior_bwd_kernel<false, 0, BOX_OVERLAP, Subset, SampFwdChl, SampBwdChl>,
      integprior_bwd_kernel<true, 0, BOX_OVERLAP, Subset, SampFwdChl, SampBwdChl>,
      integprior_bwd_kernel<false, 1, BOX_OVERLAP, Subset, SampFwdChl, SampBwdChl>,
      integprior_bwd_kernel<true, 1, BOX_OVERLAP, Subset, SampFwdChl, SampBwdChl>,
      integprior_bwd_kernel<false, 2, BOX_OVERLAP, Subset, SampFwdChl, SampBwdChl>,
      integprior_bwd_kernel<true, 2, BOX_OVERLAP, Subset, SampFwdChl, SampBwdChl>);
  return integprior_bwd_kernel_fn_ptrs[kernel_func_index(do_warp, accum, chlast)];
}
