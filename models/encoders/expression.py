# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Expression encoder"""

from typing import Dict, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import models.utils
from models.encoders.utils import generate_geomap


class ExpressionEncoder(nn.Module):
    """Encoder for a person's expression (as opposed to identity)
    The encoder and its KL divergence loss make the expression space smooth and consistent across different captures.
    We discard this encoder after training, and plug other ways to drive the decoder, eg, from a head-mounted device
    with outside-in cameras, such as a Quest Pro.
    """

    def __init__(
        self,
        uv_tidx: Union[torch.Tensor, np.ndarray],
        uv_bary: Union[torch.Tensor, np.ndarray],
        encoder_channel_mult: int = 1,
    ):
        """
        TODO(julieta) document args
        """

        super(ExpressionEncoder, self).__init__()

        # Convert to torch.tensor if the arrays come in np format
        uv_tidx = torch.from_numpy(uv_tidx) if type(uv_tidx) == np.ndarray else uv_tidx
        uv_bary = torch.from_numpy(uv_bary) if type(uv_bary) == np.ndarray else uv_bary
        self.register_buffer("uv_tidx", uv_tidx.type(torch.LongTensor))
        self.register_buffer("uv_bary", uv_bary.type(torch.FloatTensor))

        self.C = encoder_channel_mult
        C = self.C

        c = models.utils.Conv2dWN
        a = lambda: nn.LeakyReLU(0.2, inplace=True)
        s = nn.Sequential

        # fmt: off
        # Texture pre-processor
        #                                         in-shape, out-shape
        self.tex = s(c(    3,  16*C, 4, 2, 1), a(), # 1024, 512
                     c( 16*C,  32*C, 4, 2, 1), a(), # 512, 256
                     c( 32*C,  64*C, 4, 2, 1), a(), # 256, 128
                    )

        # Geometry pre-processor
        self.geo = s(c(    3, 16*C, 4, 2, 1), a(), # 1024, 512
                     c( 16*C, 32*C, 4, 2, 1), a(), # 512, 256
                     c( 32*C, 32*C, 4, 2, 1), a(), # 256, 128
                    )

        # Texture/Geometry combiner
        self.comb = s(c((64+32)*C,  128*C, 4, 2, 1), a(), # 128, 64
                      c(    128*C,  256*C, 4, 2, 1), a(), # 64, 32
                      c(    256*C,  256*C, 4, 2, 1), a(), # 32, 16
                      c(    256*C,  512*C, 4, 2, 1), a(), # 16, 8
                      c(    512*C,  256*C, 3, 1, 1), a(), # no change in res
                      c(    256*C,  128*C, 3, 1, 1), a(), # no change in res
                      c(    128*C,   64*C, 3, 1, 1), a(), # no change in res
                      c(     64*C,     64, 4, 2, 1), a(), # 8, 4
                    )
        # fmt: on

        # Initialize the weights
        models.utils.initseq(self.tex)
        models.utils.initseq(self.geo)
        models.utils.initseq(self.comb)

    def forward(
        self,
        verts: torch.Tensor,
        avgtex: torch.Tensor,
        neut_verts: torch.Tensor,
        neut_avgtex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            verts: [N, 3] vertices of the expression to encode
            avgtex: [3, H, W] unwrapped average texture of the expression to encode
            neut_verts: [N, 3] vertices of an average expression
            neut_avgtex: [3, H, W] unwrapped average texture of neutral expression
        Returns:
            A [64, 4, 4] code representing the expression
        """

        geo = self.geo(generate_geomap(verts - neut_verts, self.uv_tidx, self.uv_bary))
        tex = self.tex(avgtex - neut_avgtex)
        expr_code = self.comb(torch.cat((tex, geo), dim=1))
        return expr_code
