from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import models.utils
from models.encoders.utils import generate_geomap


class IdentityEncoder(nn.Module):
    """Encodes a person's identity"""

    def __init__(
        self,
        uv_tidx: Union[torch.Tensor, np.ndarray],
        uv_bary: Union[torch.Tensor, np.ndarray],
        wsize=128,
    ):
        """TODO(julieta) document params, specially warpsize and its effects"""
        super(IdentityEncoder, self).__init__()

        # Convert to torch.tensor if the arrays come in np format
        uv_tidx = torch.from_numpy(uv_tidx) if type(uv_tidx) == np.ndarray else uv_tidx
        uv_bary = torch.from_numpy(uv_bary) if type(uv_bary) == np.ndarray else uv_bary
        self.register_buffer("uv_tidx", uv_tidx.type(torch.LongTensor))
        self.register_buffer("uv_bary", uv_bary.type(torch.FloatTensor))

        self.tex = UnetEncoder()
        self.geo = UnetEncoder()
        self.comb = GeoTexCombiner()

        self.wsize = wsize
        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, wsize), np.linspace(-1.0, 1.0, wsize))
        grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpidentity", torch.from_numpy(grid))
        self.bias = nn.Parameter(torch.zeros(1, 2, wsize, wsize))
        self.bias.data.zero_()

    def forward(
        self, neut_verts: torch.Tensor, neut_avgtex: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        geo = generate_geomap(neut_verts, self.uv_tidx, self.uv_bary)
        z_geo, b_geo = self.geo(geo)
        z_tex, b_tex = self.tex(neut_avgtex)
        b_geo, b_tex = self.comb(b_geo, b_tex)

        warp = self.warpidentity + self.bias / self.wsize
        for i in range(len(b_geo)):
            w, h, b = b_geo[i].shape[-1], b_geo[i].shape[-2], b_geo[i].shape[0]
            W = (
                torch.nn.functional.interpolate(warp, size=(h, w), mode="bilinear")
                .permute(0, 2, 3, 1)
                .repeat(b, 1, 1, 1)
            )
            b_geo[i] = torch.nn.functional.grid_sample(b_geo[i], W, align_corners=False)
            b_tex[i] = torch.nn.functional.grid_sample(b_tex[i], W, align_corners=False)

        return {"z_geo": z_geo, "z_tex": z_tex, "b_geo": b_geo, "b_tex": b_tex}


class UnetEncoder(nn.Module):
    """Encoder of a UNet that outputs a list of bias maps"""

    def __init__(self, imsize: int = 1024, channel_mult: int = 1, input_chan: int = 3):
        super(UnetEncoder, self).__init__()

        self.imsize = imsize
        l = models.utils.LinearWN
        c = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential
        C = channel_mult

        self.layers = nn.ModuleDict()

        if imsize == 1024:
            esize = [input_chan, 16 * C, 32 * C, 64 * C, 64 * C, 128 * C, 128 * C, 256 * C, 256 * C]
            bsize = [input_chan, 16, 32, 64, 64, 128, 128, 256, 256]
        else:
            print(f"Unsupported image size: {imsize}")
            quit()
        self.nlayers = len(esize) - 1
        for i in range(self.nlayers):
            e: List[nn.Module] = [c(esize[i], esize[i + 1], 4, 2, 1)]
            b: List[nn.Module] = [c(esize[i], bsize[i], 1, 1, 0)]
            e.append(a(0.2, inplace=True))
            if i > 0:
                b.append(a(0.2, inplace=True))
            self.layers[f"e{i}"] = s(*e)
            self.layers[f"b{i}"] = s(*b)
        self.enc = c(esize[-1], 16, 1, 1, 0)

        for i in range(self.nlayers):
            models.utils.initseq(self.layers[f"e{i}"])
            models.utils.initseq(self.layers[f"b{i}"])
        models.utils.initmod(self.enc)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        n, b = x.shape[0], []
        for i in range(self.nlayers):
            # skip first one since not used?
            # bi = None if i == 0 else self._modules[f'b{i}'](x)
            bi = self.layers[f"b{i}"](x)
            b.insert(0, bi)
            x = self.layers[f"e{i}"](x)
        z = self.enc(x)

        return z, b


class GeoTexCombiner(nn.Module):
    """Module that mixes information from both geometry and texture bias maps"""

    def __init__(self, imsize: int = 1024, input_chan: int = 3):
        super(GeoTexCombiner, self).__init__()

        self.imsize = imsize
        if self.imsize == 1024:
            # The number of channels of the input texture bias maps
            tsize = [input_chan, 16, 32, 64, 64, 128, 128, 256]
            gsize = [input_chan, 16, 32, 64, 64, 128, 128, 256]
        else:
            raise ValueError(f"Unsupported image size: {self.imsize}")

        self.layers = nn.ModuleDict()

        # TODO(julieta) this is traversing the sizes backwards. Consider doing that beforehand
        n = len(gsize)
        for i in range(n):
            sg, st = gsize[-1 - i], tsize[-1 - i]

            t2g = nn.Sequential(models.utils.Conv2dWN(st, sg, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))
            g = nn.Sequential(models.utils.Conv2dWN(sg * 2, sg, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))

            g2t = nn.Sequential(models.utils.Conv2dWN(sg, st, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))
            t = nn.Sequential(models.utils.Conv2dWN(st * 2, st, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))

            self.layers[f"t2g{i}"] = t2g
            self.layers[f"g2t{i}"] = g2t
            self.layers[f"g{i}"] = g
            self.layers[f"t{i}"] = t

            models.utils.initseq(self.layers[f"g{i}"])
            models.utils.initseq(self.layers[f"t{i}"])
            models.utils.initseq(self.layers[f"t2g{i}"])
            models.utils.initseq(self.layers[f"g2t{i}"])

    def forward(self, b_geo_id: List[torch.Tensor], b_tex_id: List[torch.Tensor]):
        for i in range(len(b_geo_id)):
            cg = torch.cat([b_geo_id[i], self.layers[f"t2g{i}"](b_tex_id[i])], dim=1)
            ct = torch.cat([b_tex_id[i], self.layers[f"g2t{i}"](b_geo_id[i])], dim=1)
            b_geo_id[i] = self.layers[f"g{i}"](cg)
            b_tex_id[i] = self.layers[f"t{i}"](ct)

        return b_geo_id, b_tex_id
