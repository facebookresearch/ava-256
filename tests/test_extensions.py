# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import einops
import numpy as np
import pytest
import torch

from extensions.utils.utils import compute_raydirs
from models.raymarchers.mvpraymarcher import Raymarcher as RaymarcherOSS
from utils import load_camera_calibration


@pytest.fixture
def imshape() -> Tuple[int, int]:
    imwidth, imheight = 1334, 2048
    return imwidth, imheight


@pytest.fixture
def KRT() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    krt_dicts = load_camera_calibration("assets/camera_calibration.json")
    krt = krt_dicts["400940"]
    downsample = 1.0

    campos = torch.from_numpy((-np.dot(krt["extrin"][:3, :3].T, krt["extrin"][:3, 3])).astype(np.float32)).cuda()
    camrot = torch.from_numpy((krt["extrin"][:3, :3]).astype(np.float32)).cuda()
    focal = torch.from_numpy((np.diag(krt["intrin"][:2, :2]) / downsample).astype(np.float32)).cuda()
    princpt = torch.from_numpy((krt["intrin"][:2, 2] / downsample).astype(np.float32)).cuda()

    campos = campos[None, :]
    camrot = camrot[None, :]
    focal = focal[None, :]
    princpt = princpt[None, :]

    return campos, camrot, focal, princpt


@pytest.fixture
def raydirs(KRT, imshape):
    """Implicitly test compute_raydirs from OSS extensions"""
    campos, camrot, focal, princpt = KRT
    campos = campos.contiguous()
    camrot = camrot.contiguous()
    focal = focal.contiguous()
    princpt = princpt.contiguous()

    imwidth, imheight = imshape

    volume_radius = 256.0

    px, py = np.meshgrid(np.arange(imwidth).astype(np.float32), np.arange(imheight).astype(np.float32))
    pixelcoords = torch.from_numpy(np.stack((px, py), axis=-1))[None, :].cuda()

    raypos, raydir, tminmax = compute_raydirs(campos, camrot, focal, princpt, pixelcoords, volume_radius)

    assert raypos.shape == torch.Size([1, imheight, imwidth, 3])
    assert raydir.shape == torch.Size([1, imheight, imwidth, 3])
    assert tminmax.shape == torch.Size([1, imheight, imwidth, 2])

    return raypos, raydir, tminmax


def test_extension_shapes(raydirs, imshape):
    """Check the shapes of the raymarcher, and compare OSS and internal version"""

    raypos, raydir, tminmax = raydirs
    imwidth, imheight = imshape

    # Load a dummy geometry object to get plausible mean and std verts
    verts = torch.from_numpy(np.fromfile("assets/021924.bin", dtype=np.float32).reshape(1, -1, 3))
    primsize = (8, 8, 8)

    decout = {
        "verts": verts.cuda(),
        "template": torch.rand(1, 128**2, 4, *primsize).cuda(),
        "primpos": torch.rand(1, 128**2, 3).cuda(),
        "primrot": torch.rand(1, 128**2, 3, 3).cuda(),
        "primscale": torch.rand(1, 128**2, 3).cuda(),
    }

    volradius = 256.0
    raymarcher = RaymarcherOSS(volradius)

    ##### OSS check...
    # OSS implementation, template takes channels last
    decout["template"] = einops.rearrange(decout["template"], "n k c td th tw -> n k td th tw c").contiguous()

    rayrgb, rayalpha, _, pos_img = raymarcher(
        raypos,
        raydir,
        tminmax,
        decout,
    )

    assert rayrgb.shape == torch.Size([1, 3, imheight, imwidth])
    assert rayalpha.shape == torch.Size([1, 1, imheight, imwidth])
    assert pos_img is None
