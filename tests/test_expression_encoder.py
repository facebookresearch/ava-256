# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import einops
import numpy as np
import torch
from PIL import Image

from models.encoders.expression import ExpressionEncoder
from utils import create_uv_baridx, load_obj, make_closest_uv_barys


def test_encoder_sizes():
    """Smoke test confirming expected sizes for the encoder"""
    # TODO(julieta) make a centralized file with asset names so we don't have to type them out every time

    # Load topology
    uvdata = create_uv_baridx("assets/face_topology.obj", resolution=1024)

    # Load real verts, create dummy avg verts
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))
    neut_verts = verts + torch.rand(*verts.shape, dtype=torch.float32)

    # Load textures
    avgtex = np.array(Image.open("assets/021924_avgtex.png")).astype(np.float32)
    avgtex = torch.from_numpy(einops.rearrange(avgtex, "H W C -> 1 C H W"))
    neut_avgtex = avgtex + torch.rand(*avgtex.shape, dtype=torch.float32)

    # Create expression encoder
    expression_encoder = ExpressionEncoder(uvdata["uv_idx"], uvdata["uv_bary"])
    expr_code = expression_encoder(verts, avgtex, neut_verts, neut_avgtex)

    assert expr_code.shape == torch.Size([1, 64, 4, 4])
