# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Raymarcher for a mixture of volumetric primitives """
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.mvpraymarch.mvpraymarch import mvpraymarch


class Raymarcher(nn.Module):
    def __init__(self, volradius, dt: float = 1.0):
        super(Raymarcher, self).__init__()

        self.volume_radius = volradius

        # step size
        self.dt = dt / self.volume_radius

    def forward(
        self,
        raypos: torch.Tensor,
        raydir: torch.Tensor,
        tminmax: torch.Tensor,
        decout: Dict[str, torch.Tensor],
        renderoptions={},
        rayterm=None,
        with_pos_img=None,
    ):
        rayrgba = mvpraymarch(
            raypos,
            raydir,
            self.dt,
            tminmax,
            (decout["primpos"], decout["primrot"], decout["primscale"]),
            template=decout["template"],
            warp=decout["warp"] if "warp" in decout else None,
            rayterm=rayterm,
            **{k: v for k, v in renderoptions.items() if k in mvpraymarch.__code__.co_varnames}
        )

        assert rayrgba is not None

        rayrgba = rayrgba.permute(0, 3, 1, 2)
        rayrgb, rayalpha = rayrgba[:, :3].contiguous(), rayrgba[:, 3:4].contiguous()
        pos_img = None

        return rayrgb, rayalpha, rayrgba, pos_img
