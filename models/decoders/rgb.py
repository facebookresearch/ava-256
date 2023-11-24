from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

import models.utils


class RGBDecoder(nn.Module):
    """Decoder for RGB values of each volumetric primitive"""

    def __init__(
        self,
        *,
        imsize: int,
        nboxes: int,
        boxsize: int,
        outch: int,
        viewcond: bool = True,
    ):
        """
        TODO(julieta) document args
        """
        super(RGBDecoder, self).__init__()

        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize
        self.outch = outch
        self.viewcond = viewcond

        self.layers = nn.ModuleDict()

        nh = int(np.sqrt(self.nboxes))
        assert nh * nh == self.nboxes
        if nh == 512:
            assert boxsize == 2
        elif nh == 64:
            assert boxsize == 16
        elif nh == 128:
            assert boxsize == 8
        else:
            print(f"boxsize {boxsize} not supported yet")

        l = models.utils.LinearWN
        c = models.utils.ConvTranspose2dWNUB
        v = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential

        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        inch = 16 + 16
        if self.viewcond:
            self.viewmod = s(l(3, 16), a(0.2, inplace=True), l(16, 8 * 4 * 4), a(0.2, inplace=True))
            inch += 8

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize * self.outch]
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize * self.outch]
        else:
            raise ValueError(f"Unsupported image size: {imsize}")

        self.nlayers = len(size) - 1

        h = 8
        for i in range(self.nlayers):
            t: List[nn.Module] = [c(size[i], size[i + 1], h, h, 4, 2, 1)]
            h *= 2

            if i < self.nlayers - 1:
                t.append(a(0.2, inplace=True))
            self.layers[f"t{i}"] = s(*t)

        if self.viewcond:
            models.utils.initseq(self.viewmod)
        models.utils.initseq(self.encmod)
        for i in range(self.nlayers):
            models.utils.initseq(self.layers[f"t{i}"])

        self.bias = nn.Parameter(torch.zeros(self.boxsize * self.outch, imsize, imsize))
        self.bias.data.zero_()

    def forward(
        self,
        ex_enc: torch.Tensor,
        id_enc: torch.Tensor,
        id_biases: List[torch.Tensor],
        view: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            ex_enc: [N, 16, 4, 4] expression code
            id_enc: [N, 16, 4, 4] identity code
            id_bias: List of identity biases
            view: [N, 3] View direction that the decoder might use to model view-dependent effects
        """

        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        if self.viewcond:
            v = self.viewmod(view).view(-1, 8, 4, 4)
            x = torch.cat([v, x], dim=1)

        scale = 1 / np.sqrt(2)
        for i in range(self.nlayers):
            xx = self.layers[f"t{i}"](x)

            if id_biases is not None:
                n = id_biases[i].shape[1]

                if n == xx.shape[1]:
                    x = (xx + id_biases[i]) * scale
                else:
                    x = xx  # note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

        tex = x + self.bias[None, :, :, :]

        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)

        # TODO(julieta) document this transform, rewrite with einops
        x = x.view(x.size(0), self.boxsize, self.outch, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)
        x = x.reshape(x.size(0), self.nboxes, self.outch, self.boxsize, self.boxsize, self.boxsize)

        return x
