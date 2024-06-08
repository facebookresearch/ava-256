# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def generate_geomap(geo: torch.Tensor, uv_tidx: torch.Tensor, uv_bary: torch.Tensor) -> torch.Tensor:
    """
    Create a geometry image given a series of vertices and their topology
    Args:
        geo: [1, N, 3] tensor with N vertices
        uv_tidx: [3, M, M] tensor with geometry image indices
        uv_bary: [3, M, M] tensor with geometry image barycentric coordinates
    Returns:
      An [M, M, 2] tensor representing the input geometry as an image

    NOTE(julieta) if the source topology of `uv_tidx` and `uv_bary` is low res but the images themselves are high res,
      `uv_tidx` will contain multiple repeated indices, which means that backpropagating through this function would be
      very very slow (see https://github.com/pytorch/pytorch/issues/41162#issuecomment-655834491).
    """

    # NOTE(julieta) this function gets called a couple times per iteration, so asserts can add up. Disabling for speed.
    # assert geo.ndim == 3
    # assert uv_tidx.ndim == 3
    # assert uv_bary.ndim == 3

    # assert geo.shape[2] == 3
    # assert uv_tidx.shape[1] == uv_tidx.shape[2]
    # assert uv_tidx.shape[2] == uv_bary.shape[2]

    n = geo.shape[0]
    g = geo.view(n, -1, 3).permute(0, 2, 1)

    geomap = (
        g[:, :, uv_tidx[0]] * uv_bary[0][None, None, :, :]
        + g[:, :, uv_tidx[1]] * uv_bary[1][None, None, :, :]
        + g[:, :, uv_tidx[2]] * uv_bary[2][None, None, :, :]
    )

    return geomap
