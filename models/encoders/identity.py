from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils


class EncoderIdentity(nn.Module):
    """Encodes a person's identity"""

    def __init__(self, uv_tidx: np.ndarray, uv_bary: np.ndarray, wsize=128):
        super(EncoderIdentity, self).__init__()

        self.register_buffer("uv_tidx", torch.from_numpy(uv_tidx).type(torch.LongTensor))
        self.register_buffer("uv_bary", torch.from_numpy(uv_bary).type(torch.FloatTensor))

        self.tex = EncoderUNet()
        self.geo = EncoderUNet()
        self.comb = GeoTexCombiner()

        self.wsize = wsize
        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, wsize), np.linspace(-1.0, 1.0, wsize))
        grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpidentity", torch.from_numpy(grid))
        self.bias = nn.Parameter(torch.zeros(1, 2, wsize, wsize))
        self.bias.data.zero_()

    def forward(self, neut_verts, neut_avgtex, losslist=None):
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

        return {"z_geo": z_geo, "z_tex": z_tex, "b_geo": b_geo, "b_tex": b_tex}, None


###############################################################################
class EncoderUNet(nn.Module):
    def __init__(self, ncond=1, imsize=1024, channel_mult=1, input_chan=3):
        super(EncoderUNet, self).__init__()

        self.ncond = ncond
        self.imsize = imsize
        l = models.utils.LinearWN
        c = models.utils.Conv2dWN
        # a = CenteredLeakyReLU
        a = nn.LeakyReLU
        s = nn.Sequential
        C = channel_mult

        if imsize == 1024:
            esize = [input_chan * ncond, 16 * C, 32 * C, 64 * C, 64 * C, 128 * C, 128 * C, 256 * C, 256 * C]
            bsize = [input_chan, 16, 32, 64, 64, 128, 128, 256, 256]
        else:
            print(f"Unsupported image size: {imsize}")
            quit()
        self.nlayers = len(esize) - 1
        for i in range(self.nlayers):
            e = [c(esize[i], esize[i + 1], 4, 2, 1)]
            b = [c(esize[i], bsize[i], 1, 1, 0)]
            e.append(a(0.2, inplace=True))
            if i > 0:
                b.append(a(0.2, inplace=True))
            self.add_module(f"e{i}", s(*e))
            self.add_module(f"b{i}", s(*b))
        self.enc = c(esize[-1], 16, 1, 1, 0)

        for i in range(self.nlayers):
            models.utils.initseq(self._modules[f"e{i}"])
            models.utils.initseq(self._modules[f"b{i}"])
        models.utils.initmod(self.enc)

    def forward(self, x):
        #############################
        x_orig = x
        #############################

        n, b = x.shape[0], []
        for i in range(self.nlayers):
            # skip first one since not used?
            # bi = None if i == 0 else self._modules[f'b{i}'](x)
            bi = self._modules[f"b{i}"](x)
            b.insert(0, bi)
            x = self._modules[f"e{i}"](x)
        z = self.enc(x)

        #######################################################################################
        if not np.isfinite(torch.sum(z).item()):
            print("------------------- Non Finite Encoding --------------------------")
            x = torch.sum(x_orig.contiguous().view(n, -1), dim=-1)
            print(f"x: {x}")

            z = torch.sum(z.contiguous().view(n, -1), dim=-1)
            print(f"z: {z}")

            for i in range(len(b)):
                bi = torch.sum(b[i].contiguous().view(n, -1), dim=-1)
                print(f"b{i}: {bi}")

            for i in range(len(b)):
                wi = torch.sum(self._modules[f"e{i}"][0].weight).item()
                bi = torch.sum(self._modules[f"e{i}"][0].bias).item()
                print(f"d{i}: {wi} {bi}")

            for i in range(len(b)):
                wi = torch.sum(self._modules[f"b{i}"][0].weight).item()
                bi = torch.sum(self._modules[f"b{i}"][0].weight).item()
                print(f"u{i}: {wi} {bi}")

            quit()
        #######################################################################################

        return z, b


###############################################################################
class GeoTexCombiner(nn.Module):
    def __init__(self, texsize=1024, geosize=1024, input_chan=3):
        super(GeoTexCombiner, self).__init__()

        self.texsize, self.geosize = texsize, geosize

        if self.texsize == 1024:
            tsize = [input_chan, 16, 32, 64, 64, 128, 128, 256]
        elif self.texsize == 512:
            tsize = [input_chan, 16, 32, 64, 64, 128, 128]
        elif self.texsize == 256:
            tsize = [input_chan, 16, 32, 64, 64, 128]
        elif self.texsize == 128:
            tsize = [input_chan, 16, 32, 64, 64]

        if self.geosize == 1024:
            gsize = [input_chan, 16, 32, 64, 64, 128, 128, 256]
        elif self.geosize == 512:
            gsize = [input_chan, 16, 32, 64, 64, 128, 128]
        elif self.geosize == 256:
            gsize = [input_chan, 16, 32, 64, 64, 128]
        elif self.geosize == 128:
            gsize = [input_chan, 16, 32, 64, 64]

        n = len(gsize)
        for i in range(n):
            sg, st = gsize[-1 - i], tsize[-1 - i]

            t2g = nn.Sequential(models.utils.Conv2dWN(st, sg, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))
            g = nn.Sequential(models.utils.Conv2dWN(sg * 2, sg, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))

            g2t = nn.Sequential(models.utils.Conv2dWN(sg, st, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))
            t = nn.Sequential(models.utils.Conv2dWN(st * 2, st, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True))

            self.add_module(f"t2g{i}", t2g)
            self.add_module(f"g2t{i}", g2t)
            self.add_module(f"g{i}", g)
            self.add_module(f"t{i}", t)

            models.utils.initseq(self._modules[f"g{i}"])
            models.utils.initseq(self._modules[f"t{i}"])
            models.utils.initseq(self._modules[f"t2g{i}"])
            models.utils.initseq(self._modules[f"g2t{i}"])

    def forward(self, b_geo_id, b_tex_id):
        for i in range(len(b_geo_id)):
            cg = torch.cat([b_geo_id[i], self._modules[f"t2g{i}"](b_tex_id[i])], dim=1)
            ct = torch.cat([b_tex_id[i], self._modules[f"g2t{i}"](b_geo_id[i])], dim=1)
            b_geo_id[i] = self._modules[f"g{i}"](cg)
            b_tex_id[i] = self._modules[f"t{i}"](ct)

        return b_geo_id, b_tex_id
