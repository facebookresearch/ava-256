# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from models.decoders.geometry import GeometryDecoder
from utils import load_obj


def test_geometry_decoder_sizes():
    """Ensure the sizes of the rgb decoder make sense"""

    nprims = 128 * 128
    primsize = (8, 8, 8)
    imsize = int(np.sqrt(nprims)) * primsize[1]

    # Load topology
    dotobj = load_obj("assets/face_topology.obj")
    vt, vi, vti = dotobj["vt"], dotobj["vi"], dotobj["vti"]

    # Load a dummy geometry object to get the number of vertices
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))

    decoder = GeometryDecoder(
        uv=np.array(vt),
        tri=np.array(vi),
        uvtri=np.array(vti),
        nvtx=verts.shape[-2],
        motion_size={256: 16, 16384: 128}[nprims],
        geo_size=256,
        imsize=imsize,
        nboxes=nprims,
        boxsize=primsize[0],
    )

    ex_enc = torch.rand([1, 16, 4, 4])
    id_enc = torch.rand([1, 16, 4, 4])
    id_biases = []

    # fmt: off
    bias_channels = [256, 128, 128, 64,  64,  32,  16,    3]
    bias_shapes =   [8,    16,  32, 64, 128, 256, 512, 1024]
    for c, h in zip(bias_channels, bias_shapes):
        id_biases.append(torch.rand([1, c, h, h]))
    # fmt: on

    decout, geo, primitive_position_residuals, primitive_rotation_residuals, primitive_scale_residuals = decoder(
        ex_enc, id_enc, id_biases
    )

    assert decout.shape == torch.Size([1, nprims, *primsize, 1])
    assert geo.shape == verts.shape
    assert primitive_position_residuals.shape == torch.Size([1, nprims, 3])
    assert primitive_rotation_residuals.shape == torch.Size([1, nprims, 3])
    assert primitive_scale_residuals.shape == torch.Size([1, nprims, 3])
