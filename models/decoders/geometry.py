from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils


class GeometryDecoder(nn.Module):
    def __init__(
        self,
        uv: np.ndarray,
        tri: np.ndarray,
        uvtri: np.ndarray,
        *,
        nvtx: int,
        motion_size: int,
        geo_size: int,
        imsize: int,
        nboxes: int,
        boxsize: int,
    ):
        """
        TODO(julieta) document params
        """
        super(GeometryDecoder, self).__init__()

        assert motion_size < imsize
        assert geo_size < imsize

        self.motion_size = motion_size
        self.geo_size = geo_size
        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize

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

        c = models.utils.ConvTranspose2dWNUB
        v = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential

        # reduce noise effect of latent expression code
        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        models.utils.initseq(self.encmod)

        inch = 16 + 16  # first is for expression, second for identity

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize]
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize]
        else:
            raise ValueError(f"Unsupported image size: {imsize}")

        self.layers = nn.ModuleDict()
        self.nlayers = len(size) - 1

        # build deconv arch with early exists for geometry and motion
        h = 8
        for i in range(self.nlayers):
            t: List[nn.Module] = [c(size[i], size[i + 1], h, h, 4, 2, 1)]
            if i < self.nlayers - 1:
                t.append(a(0.2, inplace=True))

            self.layers[f"t{i}"] = s(*t)
            models.utils.initseq(self.layers[f"t{i}"])

            if h == motion_size:
                self.motion = s(v(size[i + 1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 9, 1, 1, 0))
                models.utils.initseq(self.motion)

            if h == geo_size:
                self.geo = s(v(size[i + 1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 3, 1, 1, 0))
                models.utils.initseq(self.geo)

            h *= 2

        self.bias = nn.Parameter(torch.zeros(self.boxsize, imsize, imsize))
        self.bias.data.zero_()

        # create cropping coordinates for geometry points
        vlists = [list() for _ in range(nvtx)]

        for fi in range(tri.shape[0]):
            for fv in range(3):
                vlists[tri[fi, fv]].append(uvtri[fi, fv])

        nMaxUVsPerVertex = 1
        uvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.int32)
        uvmask = np.ones((nvtx,), dtype=np.float32)
        for tvi in range(len(vlists)):
            if not (len(vlists[tvi])):
                uvmask[tvi] = 0
                continue
            for vsi in range(nMaxUVsPerVertex):
                if vsi < len(vlists[tvi]):
                    uvspervert[tvi, vsi] = vlists[tvi][vsi]
                elif len(vlists[tvi]):
                    uvspervert[tvi, vsi] = vlists[tvi][0]
        t_nl_geom_vert_uvs = torch.from_numpy(uv)[uvspervert, :]
        coords = t_nl_geom_vert_uvs.view(1, -1, nMaxUVsPerVertex, 2) * 2 - 1.0
        self.register_buffer("coords", coords)

    def forward(
        self,
        ex_enc: torch.Tensor,
        id_enc: torch.Tensor,
        id_bias: List[torch.Tensor],
    ):
        """
        Args:
            ex_enc: [N, 16, 4, 4] expression code
            id_enc: [N, 16, 4, 4] identity code
            id_bias: List of identity biases
        """

        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        mot = None
        geo = None

        scale = 1 / np.sqrt(2)
        for i in range(self.nlayers):
            xx = self.layers[f"t{i}"](x)

            if id_bias is not None:
                n = id_bias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_bias[i][:, :n, ...] * 0.1 + 1.0) + id_bias[i][:, n:, ...]) * scale
                elif n * 2 == xx.shape[1]:
                    x = (xx + id_bias[i]) * scale
                else:
                    x = xx  # note: last layer (1024 x 1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

            if x.shape[-1] == self.motion_size:
                mot = self.motion(x)
            if x.shape[-1] == self.geo_size:
                geo = self.geo(x)

        # If after the forward pass we don't have motion or geometry something went very wrong
        if mot is None:
            raise ValueError("Motion size was never found")
        if geo is None:
            raise ValueError("geo size was never found")

        opacity = torch.exp((x + self.bias[None, :, :, :]) * 0.1)

        # get motion
        mot = mot.view(mot.size(0), 9, -1).permute(0, 2, 1).contiguous()
        primposresid = mot[:, :, 0:3] * 0.01
        primrvecresid = mot[:, :, 3:6] * 0.01
        primscaleresid = torch.exp(0.01 * mot[:, :, 6:9])

        # get geometry
        coords = self.coords.expand((geo.size(0), -1, -1, -1))
        geo = F.grid_sample(geo, coords, align_corners=False).mean(dim=3).permute(0, 2, 1)

        h = int(np.sqrt(self.nboxes))
        w = int(h)
        opacity = opacity.view(x.size(0), self.boxsize, 1, h, self.boxsize, w, self.boxsize)
        opacity = opacity.permute(0, 3, 5, 2, 1, 4, 6)
        opacity = opacity.reshape(x.size(0), self.nboxes, 1, self.boxsize, self.boxsize, self.boxsize)

        return opacity, geo, primposresid, primrvecresid, primscaleresid
