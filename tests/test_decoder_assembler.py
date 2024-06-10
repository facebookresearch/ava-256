# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import torch

from models.decoders.assembler import DecoderAssembler
from utils import create_uv_baridx


def test_decoder_assembler_sizes():
    """Ensure the sizes of the rgb decoder make sense"""

    nprims = 128 * 128
    primsize = (8, 8, 8)

    # Load topology
    objpath = "assets/face_topology.obj"
    resolution = 1024
    uvdata = create_uv_baridx(objpath, resolution)
    vt, vi, vti = uvdata["uv_coord"], uvdata["tri"], uvdata["uv_tri"]

    # Load a dummy geometry object to get plausible mean and std verts
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))

    decoder = DecoderAssembler(
        vt=np.array(vt, dtype=np.float32),
        vi=np.array(vi, dtype=np.int32),
        vti=np.array(vti, dtype=np.int32),
        idxim=uvdata["uv_idx"],
        barim=uvdata["uv_bary"],
        vertmean=verts,
        vertstd=verts,
        volradius=256.0,
        nprims=nprims,
        primsize=primsize,
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

    view = torch.rand(1, 3)

    id_cond = {
        "z_tex": id_enc,
        "z_geo": id_enc,
        "b_tex": id_biases,
        "b_geo": copy.deepcopy(id_biases),
    }

    decouts = decoder(id_cond, ex_enc, view)

    assert decouts["verts"].shape == verts.shape
    assert decouts["template"].shape == torch.Size([1, 128**2, *primsize, 4])
    assert decouts["primpos"].shape == torch.Size([1, 128**2, 3])
    assert decouts["primrot"].shape == torch.Size([1, 128**2, 3, 3])
    assert decouts["primscale"].shape == torch.Size([1, 128**2, 3])
