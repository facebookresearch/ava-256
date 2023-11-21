"""
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models.utils

import pyutils
import time

from math import sqrt

def getdeconvstack(nlayers, inch0, inch1, outch):
    ch0, ch1 = inch0, inch1
    layers = []
    for i in range(nlayers - 1):
        layers.append(nn.ConvTranspose3d(ch0, ch1, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        if ch0 == ch1:
            ch1 = ch0 // 2
        else:
            ch0 = ch1
    layers.append(nn.ConvTranspose3d(ch0, outch, 4, 2, 1))
    return nn.Sequential(*layers)

class DeconvContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch):
        super(DeconvContentDecoder, self).__init__()

        nlayers = int(np.log2(primsize))

        if nprims == 1:
            inch0, inch1 = 1024, 1024
        elif nprims == 8:
            inch0, inch1 = 512, 256
        elif nprims == 64:
            inch0, inch1 = 128, 128
        elif nprims == 512:
            inch0, inch1 = 64, 32
        elif nprims == 4096:
            inch0, inch1 = 16, 16
        else:
            raise

        self.dec0 = nn.Sequential(nn.Linear(inch, inch0), nn.LeakyReLU(0.2))
        self.dec1 = nn.ModuleList([
            getdeconvstack(nlayers, inch0, inch1, outch)
            for i in range(nprims)])

        print("Content Decoder:", nprims, "x")
        print(self.dec1[0])

        for m in [self.dec0, *self.dec1]:
            models.utils.initseq(m)

    def forward(self, x):
        x = self.dec0(x)
        return torch.stack([self.dec1[i](x[:, :, None, None, None]) for i in range(len(self.dec1))], dim=1)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def getjointdeconvubstack(nprims, primsize, inch0, inch1, outch, ub=False):
    nvoxels = nprims * primsize ** 3
    assert nvoxels == 2097152 or nvoxels == 2097152*8 or nvoxels == 2097152 // 2 or nvoxels == 2097152 * 8 // 2

    if ub:
        layerfunc2d = lambda w, h, a, b, k, s, p: models.utils.ConvTranspose2dUB(w, h, a, b, k, s, p)
        layerfunc3d = lambda w, h, d, a, b, k, s, p: models.utils.ConvTranspose3dUB(w, h, d, a, b, k, s, p)
    else:
        layerfunc2d = lambda w, h, a, b, k, s, p: nn.ConvTranspose2d(a, b, k, s, p)
        layerfunc3d = lambda w, h, d, a, b, k, s, p: nn.ConvTranspose3d(a, b, k, s, p)

    extralayers = 0 if nvoxels == 2097152 else 1

    ch0, ch1 = inch0, inch1
    layers = []

    reshapedim = {8: (1, 2), 64: (1, 1), 256: (1, 1), 512: (1, 2), 4096: (1, 1), 16384: (1, 1), 32768: (1, 2), 262144: (1, 1)}[nprims]
    layers.append(Reshape(-1, ch0, reshapedim[0], reshapedim[1]))
    dims = (1, reshapedim[0], reshapedim[1])

    nlayers2d = {8: 1, 64: 3, 256: 4, 512: 4, 4096: 6, 16384: 7, 32768: 7, 262144: 9}[nprims]
    nlayers3d = {8: 6, 64: 5, 256: 4, 512: 4, 4096: 3, 16384: 2, 32768: 2, 262144: 1}[nprims]
    nlayers3d += extralayers

    for i in range(nlayers2d):
        layers.append(layerfunc2d(dims[2]*2, dims[1]*2, ch0, ch1, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        if ch0 == ch1:
            ch1 = ch0 // 2
        else:
            ch0 = ch1
        dims = (dims[0], dims[1] * 2, dims[2] * 2)

    reshapedim = {8: (2, 4), 64: (8, 8), 256: (16, 16), 512: (16, 32), 4096: (64, 64), 16384: (128, 128), 32768: (128, 256), 262144: (512, 512)}[nprims]
    layers.append(Reshape(-1, ch0, 1, reshapedim[0], reshapedim[1]))

    for i in range(nlayers3d - 1):
        layers.append(layerfunc3d(dims[2]*2, dims[1]*2, dims[0]*2, ch0, ch1, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        if ch0 == ch1:
            ch1 = ch0 // 2
        else:
            ch0 = ch1
        dims = (dims[0] * 2, dims[1] * 2, dims[2] * 2)
    layers.append(layerfunc3d(dims[2]*2, dims[1]*2, dims[0]*2, ch0, outch, 4, 2, 1))
    return nn.Sequential(*layers)

class JointDeconvContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch, chdiv=1, ub=False):
        super(JointDeconvContentDecoder, self).__init__()

        self.nprims = nprims
        self.primsize = primsize
        self.outch = outch

        mult = 1
        if nprims == 1:
            inch0, inch1 = 1024, 1024
        elif nprims == 8:
            inch0, inch1 = 1024, 1024
            mult = 2
        elif nprims == 64:
            inch0, inch1 = 2048, 1024
        elif nprims == 256:
            inch0, inch1 = 2048, 1024
        elif nprims == 512:
            inch0, inch1 = 2048, 1024
            mult = 2
        elif nprims == 4096:
            inch0, inch1 = 2048, 1024
        elif nprims == 16384:
            inch0, inch1 = 2048, 1024
        elif nprims == 32768:
            inch0, inch1 = 2048, 1024
            mult = 2
        elif nprims == 262144:
            inch0, inch1 = 2048, 1024
        else:
            raise
        inch0 //= chdiv
        inch1 //= chdiv

        self.dec0 = nn.Sequential(nn.Linear(inch, inch0 * mult), nn.LeakyReLU(0.2))
        self.dec1 = getjointdeconvubstack(nprims, primsize, inch0, inch1, outch, ub=ub)

        print("Content Decoder:")
        print(self.dec1)

        for m in [self.dec0, self.dec1]:
            models.utils.initseq(m)

    def forward(self, x, trainiter=None):
        x = self.dec0(x)
        x = self.dec1(x[:, :, None, None])
        if np.sqrt(self.nprims)**2 == self.nprims:
            h, w = int(np.sqrt(self.nprims)), int(np.sqrt(self.nprims))
        else:
            h = int(np.sqrt(self.nprims / 2))
            w = int(h * 2)

        x = x.view(x.size(0), x.size(1), self.primsize, h, self.primsize, w, self.primsize)
        x = x.permute(0, 3, 5, 1, 2, 4, 6)
        x = x.reshape(x.size(0), self.nprims, x.size(3), self.primsize, self.primsize, self.primsize)

        return x

def getupconvstack(nlayers, inch0, inch1, outch):
    ch0, ch1 = inch0, inch1
    layers = []
    for i in range(nlayers - 1):
        layers.append(nn.Upsample(scale_factor=2, mode='trilinear'))
        layers.append(nn.Conv3d(ch0, ch1, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2))
        if ch0 == ch1:
            ch1 = ch0 // 2
        else:
            ch0 = ch1
    layers.append(nn.Upsample(scale_factor=2, mode='trilinear'))
    layers.append(nn.Conv3d(ch0, outch, 3, 1, 1))
    return nn.Sequential(*layers)

class UpconvContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch):
        super(UpconvContentDecoder, self).__init__()

        nlayers = int(np.log2(primsize))

        if nprims == 1:
            inch0, inch1 = 2048, 1024
        elif nprims == 8:
            inch0, inch1 = 512, 512
        elif nprims == 64:
            inch0, inch1 = 256, 128
        elif nprims == 512:
            inch0, inch1 = 64, 64
        elif nprims == 4096:
            inch0, inch1 = 32, 16
        else:
            raise

        self.dec0 = nn.Sequential(nn.Linear(inch, inch0), nn.LeakyReLU(0.2))
        self.dec1 = nn.ModuleList([
            getupconvstack(nlayers, inch0, inch1, outch)
            for i in range(nprims)])

        print("Content Decoder:", nprims, "x")
        print(self.dec1[0])

        for m in [self.dec0, *self.dec1]:
            models.utils.initseq(m)

    def forward(self, x):
        x = self.dec0(x)
        return torch.stack([self.dec1[i](x[:, :, None, None, None]) for i in range(len(self.dec1))], dim=1)

class LinearContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch):
        super(LinearContentDecoder, self).__init__()

        self.outch = outch
        self.primsize = primsize

        firstch = 256

        nbases = 32

        self.dec0 = nn.Sequential(nn.Linear(inch, firstch), nn.LeakyReLU(0.2))
        self.dec1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, nbases),
                nn.Linear(nbases, primsize**3*outch))
            for i in range(nprims)])

        print("Content Decoder:", nprims, "x")
        print(self.dec1[0])

        for m in [self.dec0, *self.dec1]:
            models.utils.initseq(m)

    def forward(self, x):
        x = self.dec0(x)
        x = torch.stack([self.dec1[i](x) for i in range(len(self.dec1))], dim=1)
        y = x.view(x.size(0), x.size(1), self.outch, self.primsize, self.primsize, self.primsize)
        return y

class MLPContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch):
        super(MLPContentDecoder, self).__init__()

        innerch = 64

        self.dec2 = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(256, 2*innerch),
                nn.Linear(256, 2*innerch),
                nn.Linear(256, 2*innerch)])
            for i in range(nprims)])

        self.dec0 = nn.Sequential(nn.Linear(inch, 256), nn.LeakyReLU(0.2))
        self.dec1 = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Conv3d(innerch, innerch, 1, 1, 0), nn.LeakyReLU(0.2), nn.InstanceNorm3d(innerch)),
                nn.Sequential(nn.Conv3d(innerch, innerch, 1, 1, 0), nn.LeakyReLU(0.2), nn.InstanceNorm3d(innerch)),
                nn.Sequential(nn.Conv3d(innerch, innerch, 1, 1, 0), nn.LeakyReLU(0.2), nn.InstanceNorm3d(innerch)),
                nn.Sequential(nn.Conv3d(innerch,   3, 1, 1, 0))])
            for i in range(nprims)])

        print("Content Decoder:", nprims, "x")
        print(self.dec1[0][0])

        for m in [self.dec0]:
            models.utils.initseq(m)
        for n in self.dec1:
            for m in n:
                models.utils.initseq(m)
        for n in self.dec2:
            for m in n:
                models.utils.initmod(m)

        gridz, gridy, gridx = torch.meshgrid(
                torch.linspace(-1., 1., primsize),
                torch.linspace(-1., 1., primsize),
                torch.linspace(-1., 1., primsize))
        self.register_buffer("grid", torch.stack((gridx, gridy, gridz), dim=0)[None], persistent=False)

    def forward(self, x):
        result = []
        x = self.dec0(x)
        for i in range(nprims):
            z = self.grid
            for j in range(3):
                sa = self.dec2[i][j](x)[:, :, None, None, None]
                z = self.dec1[i][j](z) * sa[:, :sa.size(1)//2] + sa[:, sa.size(1)//2:]
            z = self.dec1[i][-1](z)
            result.append(z)
        return torch.stack(result, dim=1)

# fixed bug with warping not being used
class SimpleWarpDeconvContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch, texwarp=False, renderprims=False, wn=True, ub=True, upconv=None):
        super(SimpleWarpDeconvContentDecoder, self).__init__()

        self.nprims = nprims
        assert primsize[1]==primsize[2]
        self.primsize_w = primsize[2]
        self.primsize_h = primsize[1]
        self.primsize_d = primsize[0]
        self.outch = outch
        self.texwarp = texwarp
        self.renderprims = renderprims

        if upconv is None:
            if wn and ub:
                convmodel = models.utils.ConvTranspose2dWNUB
            elif wn and not ub:
                convmodel = lambda chin, chout, w, h, k, s, p: models.utils.ConvTranspose2dWN(chin, chout, k, s, p)
            elif not wn and ub:
                convmodel = lambda chin, chout, w, h, k, s, p: models.utils.ConvTranspose2dUB(w, h, chin, chout, k, s, p)
            elif not wn and not ub:
                convmodel = lambda chin, chout, w, h, k, s, p: nn.ConvTranspose2d(chin, chout, k, s, p)
            if wn:
                convmodelwarp = models.utils.ConvTranspose2dWN
            else:
                convmodelwarp = nn.ConvTranspose2d
        else:
            if wn and ub:
                convmodel = models.utils.Conv2dWNUB
            elif wn and not ub:
                convmodel = lambda chin, chout, w, h, k, s, p: models.utils.Conv2dWN(chin, chout, k, s, p)
            elif not wn and ub:
                convmodel = lambda chin, chout, w, h, k, s, p: models.utils.Conv2dUB(chin, chout, w, h, k, s, p)
            elif not wn and not ub:
                convmodel = lambda chin, chout, w, h, k, s, p: nn.Conv2d(chin, chout, k, s, p)
            if wn:
                convmodelwarp = models.utils.Conv2dWN
            else:
                convmodelwarp = nn.Conv2d

        nh = int(np.sqrt(self.nprims))
        nw = nh
        cm = 2
        assert nw*nh==self.nprims

        if wn:
            self.texbranch1 = nn.Sequential(models.utils.LinearWN(inch, cm*128*4*4), nn.LeakyReLU(0.2))
        else:
            self.texbranch1 = nn.Sequential(nn.Linear(inch, cm*128*4*4), nn.LeakyReLU(0.2))
        if nh==512:
            assert nh==512
            assert nw==512
            assert self.primsize_w==2
            self.texbranch2 = nn.Sequential(
                    convmodel(cm*128,cm*128,    8,    8, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm*128,cm* 64,   16,   16, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm* 64,cm* 64,   32,   32, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm* 64,cm* 32,   64,   64, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm* 32,cm* 32,  128,  128, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm* 32,cm* 16,  256,  256, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm* 16,cm* 16,  512,  512, 4, 2, 1), nn.LeakyReLU(0.2),
                    convmodel(cm* 16,     self.primsize_d*outch, 1024, 1024, 4, 2, 1))
            assert False
        elif nh==16:
            if nh==16:
                assert nw==16
                assert self.primsize_w==32
            if upconv is None:
                self.texbranch2 = nn.Sequential(
                        convmodel(cm*128,cm*128,    8,    8, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm*128,cm* 64,   16,   16, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 64,cm* 64,   32,   32, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 64,cm* 32,   64,   64, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 32,cm* 32,  128,  128, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 32,cm* 16,  256,  256, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 16,self.primsize_d*self.outch,  512,  512, 4, 2, 1))
            else:
                self.texbranch2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm*128,cm*128,    8,    8, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm*128,cm* 64,   16,   16, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 64,cm* 64,   32,   32, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 64,cm* 32,   64,   64, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 32,cm* 32,  128,  128, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 32,cm* 16,  256,  256, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 16,self.primsize_d*self.outch,  512,  512, 3, 1, 1))
        else:
            if nh==64:
                assert nw==64
                assert self.primsize_w==16
            elif nh==128:
                assert self.primsize_w==8
            if upconv is None:
                self.texbranch2 = nn.Sequential(
                        convmodel(cm*128,cm*128,    8,    8, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm*128,cm* 64,   16,   16, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 64,cm* 64,   32,   32, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 64,cm* 32,   64,   64, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 32,cm* 32,  128,  128, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 32,cm* 16,  256,  256, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 16,cm* 16,  512,  512, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodel(cm* 16,     self.primsize_d*self.outch, 1024, 1024, 4, 2, 1))
            else:
                self.texbranch2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm*128,cm*128,    8,    8, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm*128,cm* 64,   16,   16, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 64,cm* 64,   32,   32, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 64,cm* 32,   64,   64, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 32,cm* 32,  128,  128, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 32,cm* 16,  256,  256, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 16,cm* 16,  512,  512, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodel(cm* 16,self.primsize_d*self.outch,  1024,  1024, 3, 1, 1))
        print("Content Decoder:")
        print(self.texbranch2)
        for m in [self.texbranch1, self.texbranch2]:
            models.utils.initseq(m)

        if self.texwarp:
            if wn:
                self.warpbranch1 = nn.Sequential(
                        models.utils.LinearWN(inch, 256*4*4), nn.LeakyReLU(0.2))
            else:
                self.warpbranch1 = nn.Sequential(
                        nn.Linear(inch, 256*4*4), nn.LeakyReLU(0.2))

            if nh*self.primsize_w==1024:
                if upconv is None:
                    self.warpbranch2 = nn.Sequential(
                        convmodelwarp(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp(128, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp(128,  64, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp( 64,  64, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp( 64,   2, 4, 2, 1),
                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))
                else:
                    self.warpbranch2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(256, 128, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(128, 128, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(128,  64, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp( 64,  64, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp( 64,   2, 3, 1, 1),
                        nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))

                self.bias = nn.Parameter(torch.zeros(self.primsize_d*self.outch, 1024, 1024))
                self.bias.data.zero_()

                for m in [self.warpbranch1, self.warpbranch2]:
                    models.utils.initseq(m)

                xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, 1024), np.linspace(-1.0, 1.0, 1024))
                grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
                self.register_buffer("warpidentity", torch.from_numpy(grid))
            elif nh*self.primsize_w==512:
                if upconv is None:
                    self.warpbranch2 = nn.Sequential(
                        convmodelwarp(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp(128, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp(128,  64, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp( 64,  64, 4, 2, 1), nn.LeakyReLU(0.2),
                        convmodelwarp( 64,   2, 4, 2, 1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                else:
                    self.warpbranch2 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(256, 256, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(256, 128, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(128, 128, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp(128,  64, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp( 64,  64, 3, 1, 1), nn.LeakyReLU(0.2),
                        nn.Upsample(scale_factor=2, mode=upconv, align_corners=False), convmodelwarp( 64,   2, 3, 1, 1),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

                self.bias = nn.Parameter(torch.zeros(self.primsize_d*self.outch, 512, 512))
                self.bias.data.zero_()

                for m in [self.warpbranch1, self.warpbranch2]:
                    models.utils.initseq(m)

                xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, 512), np.linspace(-1.0, 1.0, 512))
                grid = np.concatenate((xgrid[None, :, :], ygrid[None, :, :]), axis=0)[None, ...].astype(np.float32)
                self.register_buffer("warpidentity", torch.from_numpy(grid))


    def forward(self, enc, viewtemplate:bool=False, trainiter=None):
        assert trainiter is not None
        x = self.texbranch1(enc)
        tex = self.texbranch2(x.view(x.size(0), -1, 4, 4))
        if self.texwarp and not viewtemplate:
            if trainiter<1000:
                tex = tex + self.bias[None, :, :, :]
            else:
                warp = self.warpbranch2(self.warpbranch1(enc).view(-1, 256, 4, 4))
                warp = warp * (1. / 1024) + self.warpidentity
                warp = warp.permute(0, 2, 3, 1)
                texo = tex
                tex = F.grid_sample(tex, warp, align_corners=False) + self.bias[None, :, :, :]

        # TODO: clean up
        if self.renderprims and self.outch==3:

            #import rscutils.cv2 as cv2

            assert(False)

            # print("@@@@@@@@@@@@@@@@@@@@@2 cv2 .. is replaced with wrapper fn in : {}".format(cv2.__file__))
            # im = cv2.imread('/mnt/captures/tsimon/devel/neurvol/uvcheckers/grid1_16_1024.png')

            # imt = torch.from_numpy(im[:,:,[2,1,0]]).to('cuda').float()
            # X = torch.arange(0,imt.shape[1])[None,:]
            # Y = torch.arange(0,imt.shape[0])[:,None]
            # checker1 = ((X+Y)%2==0).to('cuda')
            # checker2 = ((X+Y)%2==1).to('cuda')
            # imt = imt.permute((2,0,1))[None]
            # for s in range(self.primsize):
            #     if s%2==0:
            #         col = imt*(checker1[None,None].float()*1+0.0)
            #         col = (((col/255.0)**(1.8))*255-100)/25.0
            #         tex.data[:,s*3:(s+1)*3,:,:] = col
            #     else:
            #         col = imt*(checker2[None,None].float()*1+0.0)
            #         col = (((col/255.0)**(1.8))*255-100)/25.0
            #         tex.data[:,s*3:(s+1)*3,:,:] = col

        # TODO: remove?
        if self.renderprims and self.outch==1:
            tex.data[:] = 256*4
            # tex.data[:,:,:,:] = 0
            # tex.data[:,tex.shape[1]//2,:,:] = 256*3

        x0 = tex
        x = tex
        h = int(self.nprims ** 0.5)
        w = int(h)
        x = x.view(x.size(0), self.primsize_d, self.outch, h, self.primsize_h, w, self.primsize_w)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        ###################################################################################################### DEBUG
        # if False:#self.outch==3 and trainiter % 10 == 0:
        #     import cv2
        #     x0rgb = torch.nn.functional.relu(100. + 25. * x0).clamp(max=255)
        #     #t0rgb = torch.nn.functional.relu(100. + 25. * t0).clamp(max=255)
        #     ims = []
        #     for ib in [0]:#range(0, x0rgb.shape[0])
        #         for s in range(self.primsize_d):
        #             # x0rgbs = x0rgb[ib, [s,s+self.boxsize,s+self.boxsize*2], :, :]
        #             x0rgbs = x0rgb[ib, s*3:(s+1)*3, :, :]
        #             #t0rgbs = t0rgb[ib, s*3:(s+1)*3, :, :]
        #             im1 = (x0rgbs[[2,1,0],...].cpu().permute(1,2,0).data.numpy()).astype(np.uint8)
        #             #im2 = (t0rgbs[[2,1,0],...].cpu().permute(1,2,0).data.numpy()).astype(np.uint8)
        #             #im = np.concatenate((im1,im2), axis=0)
        #             #ims.append(i)m
        #             ims.append(im1)
        #             #cv2.imwrite("slice_{:02d}.png".format(s), im)
        #         ims = np.hstack(ims)
        #         cv2.imwrite("test_simplewarpdec_tex.png".format(s), ims)
        # if False:#self.outch==1 and trainiter % 10 == 0:
        #     import cv2
        #     x0rgb = torch.nn.functional.relu(x0)#.clamp(max=255)
        #     vmax = torch.amax(x0rgb)
        #     print(f'----------------------------- {vmax}')
        #     x0rgb = (x0rgb * (255. / (vmax+1e-3))).clamp(0,255)
        #     #t0rgb = torch.nn.functional.relu(100. + 25. * t0).clamp(max=255)
        #     ims = []
        #     for ib in [0]:#range(0, x0rgb.shape[0])
        #         for s in range(self.primsize_d):
        #             # x0rgbs = x0rgb[ib, [s,s+self.boxsize,s+self.boxsize*2], :, :]
        #             #x0rgbs = x0rgb[ib, s*3:(s+1)*3, :, :]
        #             x0rgbs = x0rgb[ib, s, :, :]
        #             #t0rgbs = t0rgb[ib, s*3:(s+1)*3, :, :]
        #             im1 = (x0rgbs.cpu().data.numpy()).astype(np.uint8)
        #             #im2 = (t0rgbs[[2,1,0],...].cpu().permute(1,2,0).data.numpy()).astype(np.uint8)
        #             #im = np.concatenate((im1,im2), axis=0)
        #             #ims.append(i)m
        #             ims.append(im1)
        #             #cv2.imwrite("slice_{:02d}.png".format(s), im)
        #         ims = np.hstack(ims)
        #         cv2.imwrite("test_simplewarpdec_alpha.png".format(s), ims)
        ######################################################################################################

        x = x.reshape(x.size(0), self.nprims, self.outch, self.primsize_d, self.primsize_h, self.primsize_w)
        return x

def get_dec(dectype, **kwargs):
    if dectype == "deconv":
        return DeconvContentDecoder(**kwargs)
    elif dectype == "jointdeconv":
        return JointDeconvContentDecoder(**kwargs)
    elif dectype == "jointdeconvub":
        return JointDeconvContentDecoder(**kwargs, ub=True)
    elif dectype == "upconv":
        return UpconvContentDecoder(**kwargs)
    elif dectype == "linear":
        return LinearContentDecoder(**kwargs)
    elif dectype == "mlp":
        return MLPContentDecoder(**kwargs)
    elif dectype == "simpledeconv":
        return SimpleWarpDeconvContentDecoder(**kwargs, texwarp=False)
    elif dectype == "simplewarpdeconv":
        return SimpleWarpDeconvContentDecoder(**kwargs, texwarp=True)
    elif dectype == "simplennupconv":
        return SimpleWarpDeconvContentDecoder(**kwargs, texwarp=False, upconv="nearest")
    elif dectype == "simplewarpnnupconv":
        return SimpleWarpDeconvContentDecoder(**kwargs, texwarp=True, upconv="nearest")
    elif dectype == "simplebiupconv":
        return SimpleWarpDeconvContentDecoder(**kwargs, texwarp=False, upconv="bilinear")
    elif dectype == "simplewarpbiupconv":
        return SimpleWarpDeconvContentDecoder(**kwargs, texwarp=True, upconv="bilinear")
    else:
        raise

class LinearMotionModel(nn.Module):
    def __init__(self, nprims, mode=0):
        super(LinearMotionModel, self).__init__()

        self.nprims = nprims
        self.mode = mode

        if mode == 0:
            self.primscale = nn.Parameter(0.1 * torch.randn(nprims, 3))
        elif mode == 1:
            self.primscale = nn.Sequential(
                    nn.Linear(256, 3*nprims))
            models.utils.initseq(self.primscale)
        else:
            raise
        self.primrvec = nn.Sequential(
                nn.Linear(256, 3*nprims))
        models.utils.initseq(self.primrvec)
        self.primpos = nn.Sequential(
                nn.Linear(256, 3*nprims))
        models.utils.initseq(self.primpos)

    def forward(self, encoding):
        primposresid = self.primpos(encoding).view(encoding.size(0), self.nprims, 3) * 0.01
        primrvecresid = self.primrvec(encoding).view(encoding.size(0), self.nprims, 3) * 0.01

        if self.mode == 0:
            primscaleresid = torch.exp(self.primscale[None, :, :].repeat(encoding.size(0), 1, 1))
        elif self.mode == 1:
            primscaleresid = torch.exp(0.01 * self.primscale(encoding).view(encoding.size(0), self.nprims, 3))
        return primposresid, primrvecresid, primscaleresid

def getdeconvmotionstack(nprims, inch0, inch1, outch):
    ch0, ch1 = inch0, inch1
    layers = []

    reshapedim = {8: (1, 2), 64: (1, 1), 256: (1, 1), 512: (1, 2), 4096: (1, 1), 16384: (1, 1), 32768: (1, 2), 262144: (1, 1)}[nprims]
    layers.append(Reshape(-1, ch0, reshapedim[0], reshapedim[1]))

    nlayers2d = {8: 1, 64: 3, 256: 4, 512: 4, 4096: 6, 16384: 7, 32768: 7, 262144: 9}[nprims]

    for i in range(nlayers2d - 1):
        layers.append(nn.ConvTranspose2d(ch0, ch1, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        if ch0 == ch1:
            ch1 = ch0 // 2
        else:
            ch0 = ch1

    layers.append(nn.ConvTranspose2d(ch0, outch, 4, 2, 1))
    return nn.Sequential(*layers)

class DeconvMotionModel(nn.Module):
    def __init__(self, nprims):
        super(DeconvMotionModel, self).__init__()

        self.nprims = nprims

        mult = 1
        if nprims == 1:
            inch0, inch1 = 1024, 512
        elif nprims == 8:
            inch0, inch1 = 1024, 512
            mult = 2
        elif nprims == 64:
            inch0, inch1 = 1024, 512
        elif nprims == 256:
            inch0, inch1 = 1024, 512
        elif nprims == 512:
            inch0, inch1 = 1024, 512
            mult = 2
        elif nprims == 4096:
            inch0, inch1 = 1024, 512
        elif nprims == 16384:
            inch0, inch1 = 1024, 512
        elif nprims == 32768:
            inch0, inch1 = 1024, 512
            mult = 2
        elif nprims == 262144:
            inch0, inch1 = 1024, 512
        else:
            raise

        self.dec0 = nn.Sequential(nn.Linear(256, inch0 * mult), nn.LeakyReLU(0.2))
        self.dec1 = getdeconvmotionstack(nprims, inch0, inch1, 9)

        print("Content Decoder:")
        print(self.dec1)

    def forward(self, encoding):
        out = self.dec1(self.dec0(encoding))
        out = out.view(encoding.size(0), 9, -1).permute(0, 2, 1).contiguous()

        primposresid = out[:, :, 0:3] * 0.01
        primrvecresid = out[:, :, 3:6] * 0.01
        primscaleresid = torch.exp(0.01 * out[:, :, 6:9])
        return primposresid, primrvecresid, primscaleresid

def getupconvmotionstack(nprims, inch0, inch1, outch, upmode, layermult):
    ch0, ch1 = inch0, inch1
    layers = []

    reshapedim = {8: (1, 2), 64: (1, 1), 256: (1, 1), 512: (1, 2), 4096: (1, 1), 32768: (1, 2), 262144: (1, 1)}[nprims]
    layers.append(Reshape(-1, ch0, reshapedim[0], reshapedim[1]))

    nlayers2d = {8: 1, 64: 3, 256: 4, 512: 4, 4096: 6, 32768: 7, 262144: 9}[nprims]

    for i in range(nlayers2d - 1):
        layers.append(nn.Upsample(scale_factor=2, mode=upmode, align_corners=False))
        layers.append(nn.Conv2d(ch0, ch1, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2))
        #if ch0 == ch1:
        #    ch1 = ch0 // 2
        #else:
        #    ch0 = ch1
        if layermult == 2:
            layers.append(nn.Conv2d(ch1, ch1, 3, 1, 1))
            layers.append(nn.LeakyReLU(0.2))
        if ch0 == ch1:
            ch1 = ch0 // 2
        else:
            ch0 = ch1

    layers.append(nn.Upsample(scale_factor=2, mode=upmode, align_corners=False))
    if layermult == 2:
        layers.append(nn.Conv2d(ch0, ch0, 3, 1, 1))
        layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.Conv2d(ch0, outch, 3, 1, 1))
    return nn.Sequential(*layers)

class UpconvMotionModel(nn.Module):
    def __init__(self, nprims, upmode, layermult=1):
        super(UpconvMotionModel, self).__init__()

        self.nprims = nprims

        mult = 1
        if nprims == 1:
            inch0, inch1 = 1024, 512
        elif nprims == 8:
            inch0, inch1 = 1024, 512
            mult = 2
        elif nprims == 64:
            inch0, inch1 = 1024, 512
        elif nprims == 256:
            inch0, inch1 = 1024, 512
        elif nprims == 512:
            inch0, inch1 = 1024, 512
            mult = 2
        elif nprims == 4096:
            inch0, inch1 = 1024, 512
        elif nprims == 32768:
            inch0, inch1 = 1024, 512
            mult = 2
        elif nprims == 262144:
            inch0, inch1 = 1024, 512
        else:
            raise

        self.dec0 = nn.Sequential(nn.Linear(256, inch0 * mult), nn.LeakyReLU(0.2))
        self.dec1 = getupconvmotionstack(nprims, inch0, inch1, 9, upmode=upmode, layermult=layermult)

        print("Content Decoder:")
        print(self.dec1)

    def forward(self, encoding):
        out = self.dec1(self.dec0(encoding))
        out = out.view(encoding.size(0), 9, -1).permute(0, 2, 1).contiguous()

        primposresid = out[:, :, 0:3] * 0.01
        primrvecresid = out[:, :, 3:6] * 0.01
        primscaleresid = torch.exp(0.01 * out[:, :, 6:9])
        return primposresid, primrvecresid, primscaleresid

class MLPMotionModel(nn.Module):
    def __init__(self, nprims):
        super(MLPMotionModel, self).__init__()

        self.nprims = nprims

        # TODO: AdaIn
        self.adain = nn.ModuleList([
            nn.Linear(256, 512),
            nn.Linear(256, 512),
            nn.Linear(256, 512)])
        self.net = nn.ModuleList([
            nn.Sequential(nn.Conv2d(  2, 256, 1, 1, 0), nn.LeakyReLU(0.2), nn.InstanceNorm2d(256)),
            nn.Sequential(nn.Conv2d(256, 256, 1, 1, 0), nn.LeakyReLU(0.2), nn.InstanceNorm2d(256)),
            nn.Sequential(nn.Conv2d(256, 256, 1, 1, 0), nn.LeakyReLU(0.2), nn.InstanceNorm2d(256)),
            nn.Sequential(nn.Conv2d(256,   9, 1, 1, 0))])

        for i in range(len(self.adain)):
            models.utils.initmod(self.adain[i])
        for i in range(len(self.net)):
            models.utils.initseq(self.net[i])

        if np.sqrt(self.nprims)**2 == self.nprims:
            h, w = int(np.sqrt(self.nprims)), int(np.sqrt(self.nprims))
        else:
            h = int(np.sqrt(self.nprims / 2))
            w = int(h * 2)

        gridx = torch.linspace(-1., 1., steps=w)[None, :].repeat(h, 1)
        gridy = torch.linspace(-1., 1., steps=h)[:, None].repeat(1, w)
        grid = torch.stack([gridx, gridy], dim=0)[None]
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, enc):
        x = self.grid
        for j in range(len(self.net) - 1):
            sb = self.adain[j](enc)
            x = self.net[j](x) * sb[:, :sb.size(1)//2, None, None] + sb[:, sb.size(1)//2:, None, None]
        x = self.net[-1](x)
        x = x.view(x.size(0), 9, -1).permute(0, 2, 1).contiguous()
        #return x[:, :, :3] * 0.01, x[:, :, 3:6] * 0.01, torch.exp(0.01 * x[:, :, 6:9])
        return x[:, :, :3] * 0.001, x[:, :, 3:6] * 0.001, torch.exp(0.001 * x[:, :, 6:9])

def get_motion(motiontype, nprims, **kwargs):
    if motiontype == 0:
        return LinearMotionModel(nprims, mode=motiontype)
    elif motiontype == "linearsconst":
        return LinearMotionModel(nprims, mode=0)
    elif motiontype == 1:
        return LinearMotionModel(nprims, mode=motiontype)
    elif motiontype == "linear":
        return LinearMotionModel(nprims, mode=1)
    elif motiontype == 2:
        return MLPMotionModel(nprims)
    elif motiontype == 3 or motiontype == "deconv":
        return DeconvMotionModel(nprims)
    elif motiontype == 4 or motiontype == "nnupconv":
        return UpconvMotionModel(nprims, upmode='nearest')
    elif motiontype == 5 or motiontype == "biupconv":
        return UpconvMotionModel(nprims, upmode='bilinear')
    elif motiontype == "biupconv2":
        return UpconvMotionModel(nprims, upmode='bilinear', layermult=2)
    else:
        raise

# ###############################################################################
#unet geo decoder, unet motionm headpose cond
#remove relu from opacity and add exp
class Decoder5(nn.Module):
    def __init__(self, vt, vi, vti, vertmean, vertstd, volradius, #dectype="deconv",
                 nprims=128*128, primsize=(8,8,8),
                 alphafade=False, postrainstart=0, condsize=0,
                 motiontype="deconv", warp=None, #sharedrgba=False, geown=False,
                 disable_id_encoder=False,
                 **kwargs):
        super(Decoder5, self).__init__()

        self.volradius = volradius
        self.alphafade = alphafade
        self.postrainstart = postrainstart

        self.nprims = nprims
        self.primsize = primsize
        self.motiontype = motiontype
        self.disable_id_encoder = disable_id_encoder


        self.rodrig = models.utils.Rodrigues()

        ##############################################################################
        # self.enc = nn.Sequential(
        #         nn.Linear(256 + condsize, 256), nn.LeakyReLU(0.2),
        #         nn.Linear(256,            256), nn.LeakyReLU(0.2),
        #         nn.Linear(256,            256))

        # # geotex
        # if geown:
        #     self.geobranch = nn.Sequential(models.utils.LinearWN(256, 21918))
        # else:
        #     self.geobranch = nn.Sequential(nn.Linear(256, 21918))
        # cm = 2

        # initseq = models.utils.initseq
        # initmod = models.utils.initmod

        # self.motiondec = get_motion(motiontype, nprims=nprims)

        # if sharedrgba:
        #     self.rgbadec = get_dec(dectype, nprims=nprims, primsize=primsize, inch=256+3, outch=4, **kwargs)
        # else:
        #     self.rgbdec = get_dec(dectype, nprims=nprims, primsize=primsize, inch=256+3, outch=3, **kwargs)
        #     self.alphadec = get_dec(dectype, nprims=nprims, primsize=primsize, inch=256, outch=1, **kwargs)
        #     self.rgbadec = None

        # if warp is not None:
        #     self.warpdec = get_dec(dectype, nprims=nprims, primsize=primsize, inch=256, outch=3, **kwargs)
        # else:
        #     self.warpdec = None

        # for m in [self.geobranch]:
        #     initseq(m)
        # for m in [self.enc]:
        #     initseq(m)
        #######################################################################################


        #combine encoding and identity condition
        #self.enc = nn.Sequential(models.utils.LinearWN(4*4*16*3, 256), nn.LeakyReLU(0.2))
        #models.utils.initseq(self.enc)
        #self.enc = nn.Sequential(models.utils.LinearWN(4*4*16*3, 256), CenteredLeakyReLU(0.2, True))
        #he_init(self.enc[0], 0.2)

        #guide geometry decoder
        #motion_size = {256: 16, 16384: 128}
        #self.geobranch = ConvGeoDecoder(vt, vi, vti, vertmean.shape[-2], 16+16+16, 4, 256, motion_size[nprims])
        #self.geobranch = nn.Sequential(models.utils.LinearWN(256, 256), nn.LeakyReLU(0.2),
        #                              models.utils.LinearWN(256, 256), nn.LeakyReLU(0.2),
        #                              models.utils.LinearWN(256, 21918))
        #models.utils.initseq(self.geobranch)
        #self.geobranch = nn.Sequential(models.utils.LinearWN(256, 256), CenteredLeakyReLU(0.2, True),
        #                               models.utils.LinearWN(256, 256), CenteredLeakyReLU(0.2, True),
        #                               models.utils.LinearWN(256, 21918))
        #he_init(self.geobranch[0], 0.2)
        #he_init(self.geobranch[2], 0.2)
        #he_init(self.geobranch[4], 1)

        #box-motion decoder
        #self.motiondec = get_motion(motiontype, nprims=nprims)

        #payload decoder
        #self.rgbdec = DecoderSlab(1024, self.nboxes, self.boxsize, 3, viewcond=True, texwarp=True)
        imsize = int(sqrt(nprims)) * primsize[1]
        self.rgbdec = DecoderSlab(imsize, nprims, primsize[0], 3, viewcond=True, texwarp=False, disable_id_encoder=disable_id_encoder)
        #self.alphadec = DecoderSlab(imsize, nprims, primsize[0], 1, viewcond=False, texwarp=False)
        self.geodec = DecoderGeoSlab2(vt, vi, vti, vertmean.shape[-2], {256: 16, 16384: 128}[nprims], 256, imsize, nprims, primsize[0], disable_id_encoder=disable_id_encoder)
        #self.rgbdec = get_dec("simplewarpdeconv", nprims=nprims, primsize=primsize, inch=256+3, outch=3, **kwargs) ############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.alphadec = get_dec("simplewarpdeconv", nprims=nprims, primsize=primsize, inch=256, outch=1, **kwargs)

        if warp is not None:
            self.warpdec = get_dec("simplewarpdeconv", nprims=nprims, primsize=primsize, inch=256, outch=3, **kwargs)
        else:
            self.warpdec = None

        #######################################################################################







        self.register_buffer("vt", torch.tensor(vt), persistent=False)

        self.register_buffer("vertmean", vertmean, persistent=False)
        self.vertstd = vertstd

        # apath=os.getenv('RSC_AVATAR_RSCASSET_PATH')
        apath = "/checkpoint/avatar/jinkyuk/rsc-assets"
        idximpath = f"{apath}/idxmap"
        # idximpath = "/mnt/captures/stephenlombardi/idxmap"
        self.register_buffer("idxim",
                torch.tensor(np.load("{}/retop_idxim_1024.npy".format(idximpath))).long(), persistent=False)
        self.register_buffer("tidxim",
                torch.tensor(np.load("{}/retop_tidxim_1024.npy".format(idximpath))).long(), persistent=False)
        self.register_buffer("barim",
                torch.tensor(np.load("{}/retop_barim_1024.npy".format(idximpath))), persistent=False)


        self.register_buffer("adaptwarps", 0*torch.ones(self.nprims))

    def forward(self,
                gt_geo, #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                id_cond,
                encoding,
                viewpos,
                condinput=None,
                renderoptions={},
                trainiter=-1,
                losslist=[]):
        """
        Parameters
        ----------
        encoding : torch.Tensor [B, 256]
            Encoding of current frame
        viewpos : torch.Tensor [B, 3]
            Viewing position of target camera view
        condinput : torch.Tensor [B, ?]
            Additional conditioning input (e.g., headpose)
        renderoptions : dict
            Options for rendering (e.g., rendering debug images)
        trainiter : int,
            Current training iteration
        losslist : list,
            List of losses to compute and return
        Returns
        -------
        result : dict,
            Contains predicted vertex positions, primitive contents and
            locations, scaling, and orientation, and any losses.
        """
        nprims = self.nprims
        bsize = encoding.shape[0]

        # TODO:
        if condinput is not None:
            encoding = torch.cat([encoding, condinput], dim=1)

        #encoding = self.enc(encoding)
        #z_ex = encoding
        #encoding = self.enc(torch.cat([encoding, id_cond['z_geo'], id_cond['z_tex']], dim=1).view(bsize,-1))
        #geo = self.geobranch(encoding)
        #geo = geo.view(bsize, -1, 3)
        #geo, primposresid, primrvecresid, primscaleresid = self.geobranch(torch.cat([encoding, id_cond['z_geo'], id_cond['z_tex']], dim=1))

        z_geo, b_geo = (id_cond['z_geo'], id_cond['b_geo']) if not self.disable_id_encoder else (None, None)
        primalpha, geo, primposresid, primrvecresid, primscaleresid = self.geodec(encoding, z_geo, b_geo, iternum=trainiter)


        #######################################################################################
        # if not np.isfinite(torch.sum(geo).item()):
        #     print('------------------- Non Finite Geometry --------------------------')
        #     g = torch.sum(geo.contiguous().view(bsize, -1), dim=-1)
        #     print(f"g: {g}")

        #     z = torch.sum(id_cond['z_geo'].contiguous().view(bsize, -1), dim=-1)
        #     print(f"z: {z}")

        #     for i in range(len(id_cond['b_geo'])):
        #         b = torch.sum(id_cond['b_geo'][i].contiguous().view(bsize, -1), dim=-1)
        #         print(f"b{i}: {b}")
        #     quit()
        #######################################################################################


        geo = geo * self.vertstd + self.vertmean

        ##########################################################################################################
        geo_orig = geo
        if trainiter <= self.postrainstart:
            geo = gt_geo * self.vertstd + self.vertmean
        ##########################################################################################################


        # postex = torch.stack([
        #     self.barim[:, :, 0, None] * geo[i, self.idxim[:, :, 0], :] +
        #     self.barim[:, :, 1, None] * geo[i, self.idxim[:, :, 1], :] +
        #     self.barim[:, :, 2, None] * geo[i, self.idxim[:, :, 2], :]
        #     for i in range(geo.size(0))], dim=0).permute(0, 3, 1, 2) / self.volradius
        postex = (
            self.barim[:, :, 0, None] * geo.index_select(1, self.idxim[:, :, 0].reshape(-1)).reshape(-1, 1024, 1024, 3) +
            self.barim[:, :, 1, None] * geo.index_select(1, self.idxim[:, :, 1].reshape(-1)).reshape(-1, 1024, 1024, 3) +
            self.barim[:, :, 2, None] * geo.index_select(1, self.idxim[:, :, 2].reshape(-1)).reshape(-1, 1024, 1024, 3)
        ).permute(0, 3, 1, 2) / self.volradius

        # placement of primitives on mesh
        if self.nprims == 1:
            primpos = torch.zeros((postex.size(0), 1, 3), device=postex.device)
            primscale = 2.

            v0 = torch.tensor([0., 0., 0.], device=postex.device)[None, None, None, :].repeat(postex.size(0), 1, 1, 1)
            v1 = torch.tensor([1., 0., 0.], device=postex.device)[None, None, None, :].repeat(postex.size(0), 1, 1, 1)
            v2 = torch.tensor([0., 1., 0.], device=postex.device)[None, None, None, :].repeat(postex.size(0), 1, 1, 1)
        elif self.nprims == 8:
            primpos = postex[:, :, 256::512, 128::256].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 4.

            v0 = geo[:, self.idxim[256::512, 128::256, 0], :] # vert 0?
            v1 = geo[:, self.idxim[256::512, 128::256, 1], :] # vert 1?
            v2 = geo[:, self.idxim[256::512, 128::256, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[256::512, 128::256, 0], :]
            vt1 = self.vt[self.tidxim[256::512, 128::256, 1], :]
            vt2 = self.vt[self.tidxim[256::512, 128::256, 2], :]
        elif self.nprims == 64:
            primpos = postex[:, :, 64::128, 64::128].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 8.

            v0 = geo[:, self.idxim[64::128, 64::128, 0], :] # vert 0?
            v1 = geo[:, self.idxim[64::128, 64::128, 1], :] # vert 1?
            v2 = geo[:, self.idxim[64::128, 64::128, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[64::128, 64::128, 0], :]
            vt1 = self.vt[self.tidxim[64::128, 64::128, 1], :]
            vt2 = self.vt[self.tidxim[64::128, 64::128, 2], :]
        elif self.nprims == 256:
            primpos = postex[:, :, 32::64, 32::64].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 12.

            ###################################################
            if trainiter < 100:
                with torch.no_grad():
                    centdiffx = postex[:, :, 32::64, (32+64)::64] - postex[:, :, 32::64, 32:-64:64]
                    centdiffx = torch.cat((centdiffx, centdiffx[:,:,:,-1:]), dim=3)
                    centdiffy = postex[:, :, (32+64)::64, 32::64] - postex[:, :, 32:-64:64, 32::64]
                    centdiffy = torch.cat((centdiffy, centdiffy[:,:,-1:,:]), dim=2)
                    centdiffx = centdiffx.norm(dim=1)
                    centdiffy = centdiffy.norm(dim=1)
                    centsize = torch.max(centdiffx,centdiffy)
                    centsize = torch.max(centsize,dim=0)[0].view(self.nprims)
                    warps_vec = 2/centsize
                    if self.adaptwarps.max().item()==0:
                        self.adaptwarps.data[:] = warps_vec
                    else:
                        self.adaptwarps.data[:] = self.adaptwarps*0.9 + 0.1*warps_vec
            primscale = self.adaptwarps[None,:,None] * 0.8
            ###################################################




            v0 = geo[:, self.idxim[32::64, 32::64, 0], :] # vert 0?
            v1 = geo[:, self.idxim[32::64, 32::64, 1], :] # vert 1?
            v2 = geo[:, self.idxim[32::64, 32::64, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[32::64, 32::64, 0], :]
            vt1 = self.vt[self.tidxim[32::64, 32::64, 1], :]
            vt2 = self.vt[self.tidxim[32::64, 32::64, 2], :]

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 32::64, 32::64].permute(0,2,3,1)
            vcenterdv = geodv[:, :, 32::64, 32::64].permute(0,2,3,1) # match v0 (channels last)

        elif self.nprims == 512:
            primpos = postex[:, :, 32::64, 16::32].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 16.

            v0 = geo[:, self.idxim[32::64, 16::32, 0], :] # vert 0?
            v1 = geo[:, self.idxim[32::64, 16::32, 1], :] # vert 1?
            v2 = geo[:, self.idxim[32::64, 16::32, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[32::64, 16::32, 0], :]
            vt1 = self.vt[self.tidxim[32::64, 16::32, 1], :]
            vt2 = self.vt[self.tidxim[32::64, 16::32, 2], :]
        elif self.nprims == 4096:
            primpos = postex[:, :, 8::16, 8::16].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 32.

            v0 = geo[:, self.idxim[8::16, 8::16, 0], :] # vert 0?
            v1 = geo[:, self.idxim[8::16, 8::16, 1], :] # vert 1?
            v2 = geo[:, self.idxim[8::16, 8::16, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[8::16, 8::16, 0], :]
            vt1 = self.vt[self.tidxim[8::16, 8::16, 1], :]
            vt2 = self.vt[self.tidxim[8::16, 8::16, 2], :]
        elif self.nprims == 16384:
            # primpos = postex[:, :, 4::8, 4::8].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            # primscale = 48.

            # v0 = geo[:, self.idxim[4::8, 4::8, 0], :] # vert 0?
            # v1 = geo[:, self.idxim[4::8, 4::8, 1], :] # vert 1?
            # v2 = geo[:, self.idxim[4::8, 4::8, 2], :] # vert 2?
            # vt0 = self.vt[self.tidxim[4::8, 4::8, 0], :]
            # vt1 = self.vt[self.tidxim[4::8, 4::8, 1], :]
            # vt2 = self.vt[self.tidxim[4::8, 4::8, 2], :]

            primpos = postex[:, :, 4::8, 4::8].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 48.

            ###################################################
            if trainiter < 100:
                with torch.no_grad():
                    centdiffx = postex[:, :, 4::8, (4+8)::8] - postex[:, :, 4::8, 4:-8:8]
                    centdiffx = torch.cat((centdiffx, centdiffx[:,:,:,-1:]), dim=3)
                    centdiffy = postex[:, :, (4+8)::8, 4::8] - postex[:, :, 4:-8:8, 4::8]
                    centdiffy = torch.cat((centdiffy, centdiffy[:,:,-1:,:]), dim=2)
                    centdiffx = centdiffx.norm(dim=1)
                    centdiffy = centdiffy.norm(dim=1)
                    centsize = torch.max(centdiffx,centdiffy)
                    centsize = torch.max(centsize,dim=0)[0].view(self.nprims)
                    warps_vec = 2/centsize
                    if self.adaptwarps.max().item()==0:
                        self.adaptwarps.data[:] = warps_vec
                    else:
                        self.adaptwarps.data[:] = self.adaptwarps*0.9 + 0.1*warps_vec
            primscale = self.adaptwarps[None,:,None] * 0.8
            ###################################################

            v0 = geo[:, self.idxim[4::8, 4::8, 0], :] # vert 0?
            v1 = geo[:, self.idxim[4::8, 4::8, 1], :] # vert 1?
            v2 = geo[:, self.idxim[4::8, 4::8, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[4::8, 4::8, 0], :]
            vt1 = self.vt[self.tidxim[4::8, 4::8, 1], :]
            vt2 = self.vt[self.tidxim[4::8, 4::8, 2], :]

            geodu = postex[:, :, :, 1:] - postex[:, :, :, :-1]
            geodv = postex[:, :, 1:, :] - postex[:, :, :-1, :]
            vcenterdu = geodu[:, :, 4::8, 4::8].permute(0,2,3,1)
            vcenterdv = geodv[:, :, 4::8, 4::8].permute(0,2,3,1) # match v0 (channels last)

        elif self.nprims == 32768:
            primpos = postex[:, :, 4::8, 2::4].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 64.

            v0 = geo[:, self.idxim[4::8, 2::4, 0], :] # vert 0?
            v1 = geo[:, self.idxim[4::8, 2::4, 1], :] # vert 1?
            v2 = geo[:, self.idxim[4::8, 2::4, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[4::8, 2::4, 0], :]
            vt1 = self.vt[self.tidxim[4::8, 2::4, 1], :]
            vt2 = self.vt[self.tidxim[4::8, 2::4, 2], :]
        elif self.nprims == 262144:
            primpos = postex[:, :, 1::2, 1::2].permute(0, 2, 3, 1).contiguous().view(postex.size(0), nprims, 3)
            primscale = 128.

            v0 = geo[:, self.idxim[1::2, 1::2, 0], :] # vert 0?
            v1 = geo[:, self.idxim[1::2, 1::2, 1], :] # vert 1?
            v2 = geo[:, self.idxim[1::2, 1::2, 2], :] # vert 2?
            vt0 = self.vt[self.tidxim[1::2, 1::2, 0], :]
            vt1 = self.vt[self.tidxim[1::2, 1::2, 1], :]
            vt2 = self.vt[self.tidxim[1::2, 1::2, 2], :]
        else:
            raise

        # # compute TBN matrix
        # v01 = v1 - v0
        # v02 = v2 - v0
        # vt01 = vt1 - vt0
        # vt02 = vt2 - vt0
        # f = 1. / (vt01[None, :, :, 0] * vt02[None, :, :, 1] - vt01[None, :, :, 1] * vt02[None, :, :, 0])
        # tangent = f[:, :, :, None] * torch.stack([
        #     v01[:, :, :, 0] * vt02[None, :, :, 1] - v02[:, :, :, 0] * vt01[None, :, :, 1],
        #     v01[:, :, :, 1] * vt02[None, :, :, 1] - v02[:, :, :, 1] * vt01[None, :, :, 1],
        #     v01[:, :, :, 2] * vt02[None, :, :, 1] - v02[:, :, :, 2] * vt01[None, :, :, 1]], dim=-1)
        # tangent = tangent / torch.sqrt(torch.sum(tangent ** 2, dim=-1, keepdim=True)).clamp(min=1e-5)
        # normal = torch.cross(v01, v02, dim=3)
        # normal = normal / torch.sqrt(torch.sum(normal ** 2, dim=-1, keepdim=True)).clamp(min=1e-5)
        # bitangent = torch.cross(normal, tangent, dim=3)
        # bitangent = bitangent / torch.sqrt(torch.sum(bitangent ** 2, dim=-1, keepdim=True)).clamp(min=1e-5)

        # # set orientation
        # if True:#self.flipy:
        #     primrot = torch.stack((tangent, -bitangent, normal), dim=-2)
        # else:
        #     primrot = torch.stack((tangent, bitangent, normal), dim=-2)
        # primrot = primrot.view(encoding.size(0), -1, 3, 3).contiguous().permute(0, 1, 3, 2).contiguous()




        tangent = vcenterdu
        tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-8)
        normal = torch.cross(tangent, vcenterdv, dim=3)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp(min=1e-8)
        bitangent = torch.cross(normal, tangent, dim=3)
        bitangent = bitangent / torch.norm(bitangent, dim=-1, keepdim=True).clamp(min=1e-8)
        primrot = torch.stack((tangent, bitangent, normal), dim=-2).view(encoding.size(0), -1, 3, 3).contiguous().permute(0, 1, 3, 2).contiguous()










        #primposresid, primrvecresid, primscaleresid = self.motiondec(encoding)
        if trainiter <= self.postrainstart:
        #if True: #########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            primposresid = primposresid * 0.
            primrvecresid = primrvecresid * 0.
            primscaleresid = primscaleresid * 0. + 1.
        elif trainiter <= 2 * self.postrainstart:
            weight = (2 * self.postrainstart - trainiter) / (self.postrainstart + 1)
            primposresid = primposresid * (1-weight) + weight * 0.
            primrvecresid = primrvecresid * (1-weight) + weight * 0.
            primscaleresid = primscaleresid * (1-weight) + weight * 1.
            #print(weight)
        # else:
        #     weight = 1 / (1 + trainiter - self.postrainstart)**0.25
        #     primposresid = primposresid * (1-weight) + weight * 0.
        #     primrvecresid = primrvecresid * (1-weight) + weight * 0.
        #     primscaleresid = primscaleresid * (1-weight) + weight * 1.
        #     print(weight)
        primrotresid = self.rodrig(primrvecresid.view(-1, 3)).view(encoding.size(0), nprims, 3, 3)

        primpos = primpos + primposresid
        primrot = torch.bmm(primrot.view(-1, 3, 3), primrotresid.view(-1, 3, 3)).view(encoding.size(0), nprims, 3, 3)
        primscale = primscale * primscaleresid

        viewdirs = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=1, keepdim=True))
        # if self.rgbadec is not None:
        #     # TODO: let scale, bias be an input argument?
        #     scale = torch.tensor([25., 25., 25., 1.], device=encoding.device)
        #     bias = torch.tensor([100., 100., 100., 0.], device=encoding.device)
        #     primrgba = F.relu(bias[None, None, :, None, None, None] + scale[None, None, :, None, None, None] *
        #             self.rgbadec(torch.cat([encoding, viewdirs], dim=1), trainiter=trainiter))
        # else:
        #     primrgb = F.relu(100. + 25. * self.rgbdec(torch.cat([encoding, viewdirs], dim=1), trainiter=trainiter))
        #     primalpha = F.relu(self.alphadec(encoding, trainiter=trainiter))
        # template = torch.cat([primrgb, primalpha], dim=2)

        z_tex, b_tex = (id_cond['z_tex'], id_cond['b_tex']) if not self.disable_id_encoder else (None, None)
        primrgb = self.rgbdec(encoding, z_tex, b_tex, view=viewdirs, use_warp=False, iternum=trainiter)

        #primalpha = self.alphadec(z_ex, id_cond['z_geo'], id_cond['b_geo'], view=None, use_warp=False, iternum=trainiter)
        template = torch.cat([F.relu(primrgb * 25. + 100.), F.relu(primalpha)], dim=2)

        #primrgb = F.relu(100. + 25. * self.rgbdec(torch.cat([encoding, viewdirs], dim=1), trainiter=trainiter)) ######################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #primalpha = F.relu(self.alphadec(encoding, trainiter=trainiter))
        #template = torch.cat([primrgb, primalpha], dim=2)





        if self.warpdec is not None:
            warp = self.warpdec(encoding, trainiter=trainiter) * 0.01 + torch.stack(torch.meshgrid(
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device),
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device),
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device))[::-1], dim=0)[None, None, :, :, :, :]
        else:
            warp = None

        if self.alphafade:
            gridz, gridy, gridx = torch.meshgrid(
                    torch.linspace(-1., 1., self.primsize, device=encoding.device),
                    torch.linspace(-1., 1., self.primsize, device=encoding.device),
                    torch.linspace(-1., 1., self.primsize, device=encoding.device))
            grid = torch.stack([gridx, gridy, gridz], dim=-1)
            template = template * torch.stack([
                gridx * 0. + 1.,
                gridy * 0. + 1.,
                gridz * 0. + 1.,
                torch.exp(-8.0 * torch.sum(grid ** 8, dim=-1))], dim=0)[None, None, :, :, :, :]

        if "viewaxes" in renderoptions and renderoptions["viewaxes"]:
            template[:, :, 3, template.size(3)//2:template.size(3)//2+1, template.size(4)//2:template.size(4)//2+1, :] = 2550.
            template[:, :, 0, template.size(3)//2:template.size(3)//2+1, template.size(4)//2:template.size(4)//2+1, :] = 2550.
            template[:, :, 3, template.size(3)//2:template.size(3)//2+1, :, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 1, template.size(3)//2:template.size(3)//2+1, :, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 3, :, template.size(4)//2:template.size(4)//2+1, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 2, :, template.size(4)//2:template.size(4)//2+1, template.size(5)//2:template.size(5)//2+1] = 2550.

        if "colorprims" in renderoptions and renderoptions["colorprims"]:
            lightdir = torch.tensor([1., 1., 1.], device=template.device)
            lightdir = lightdir / torch.sqrt(torch.sum(lightdir ** 2))
            zz, yy, xx = torch.meshgrid(
                torch.linspace(-1., 1., template.size(-1), device=template.device),
                torch.linspace(-1., 1., template.size(-1), device=template.device),
                torch.linspace(-1., 1., template.size(-1), device=template.device))
            primnormalx = torch.where(
                    (torch.abs(xx) >= torch.abs(yy)) & (torch.abs(xx) >= torch.abs(zz)),
                    torch.sign(xx) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormaly = torch.where(
                    (torch.abs(yy) >= torch.abs(xx)) & (torch.abs(yy) >= torch.abs(zz)),
                    torch.sign(yy) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormalz = torch.where(
                    (torch.abs(zz) >= torch.abs(xx)) & (torch.abs(zz) >= torch.abs(yy)),
                    torch.sign(zz) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormal = torch.stack([primnormalx, primnormaly, primnormalz], dim=-1)
            primnormal = primnormal / torch.sqrt(torch.sum(primnormal ** 2, dim=-1, keepdim=True))
            template[:, :, 3, :, :, :] = 1000.
            np.random.seed(123456)
            for i in range(template.size(1)):
                template[:, i, 0, :, :, :] = np.random.rand() * 255.
                template[:, i, 1, :, :, :] = np.random.rand() * 255.
                template[:, i, 2, :, :, :] = np.random.rand() * 255.

                lightdir0 = torch.sum(primrot[:, i, :, :] * lightdir[None, :, None], dim=-2)
                template[:, i, :3, :, :, :] *= 1.2 * torch.sum(
                        lightdir0[:, None, None, None, :] * primnormal, dim=-1)[:, None, :, :, :].clamp(min=0.05)

        # visualize slab
        if "viewslab" in renderoptions and renderoptions["viewslab"]:
            yy, xx = torch.meshgrid(
                    torch.linspace(0.9, -0.9, 16, device=template.device),
                    torch.linspace(-0.9, 0.9, 16, device=template.device))
            primpos = torch.stack([xx, yy, xx*0.], dim=-1)[None, :, :, :].repeat(template.size(0), 1, 1, 1).view(-1, 256, 3)
            primrot = torch.eye(3, device=template.device)[None, None, :, :].repeat(template.size(0), 256, 1, 1)
            primscale = torch.ones((template.size(0), 256, 3), device=template.device) * 16.

        losses = {}

        #if "primvol" in losslist:
        #    losses["primvol"] = torch.mean(torch.prod(1. / primscale, dim=-1), dim=-1)
        if "primvolsum" in losslist:
            losses["primvolsum"] = torch.sum(torch.prod(1. / primscale, dim=-1), dim=-1)
        #if "primtoffset" in losslist:
        #    losses["primtoffset"] = torch.mean(torch.sum(primposresid.view(primposresid.size(0), -1) ** 2, dim=-1)) * \
        #            1000. / (1000. + 0.5 * trainiter)
        #if "primroffset" in losslist:
        #    losses["primroffset"] = torch.mean(torch.sum(primrvecresid.view(primrvecresid.size(0), -1) ** 2, dim=-1)) * \
        #            1000. / (1000. + 0.5 * trainiter)


        #######################################################################################################
        if trainiter <= self.postrainstart:
            geo = geo_orig
        #######################################################################################################


        return {
                "verts": geo,
                "template": template,
                "warp": warp,
                "primpos": primpos,
                "primrot": primrot,
                "primscale": primscale}, losses


###############################################################################
#add per-pixel gains as well as bias
class DecoderSlab(nn.Module):
    def __init__(self, imsize, nboxes, boxsize, outch, viewcond=False, texwarp=False, disable_id_encoder=False):
        super(DecoderSlab, self).__init__()

        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize
        self.outch = outch
        self.texwarp = texwarp
        self.viewcond = viewcond


        nh = int(np.sqrt(self.nboxes))
        assert nh*nh==self.nboxes
        if nh==512:
            assert boxsize==2
        elif nh==64:
            assert boxsize==16
        elif nh==128:
            assert boxsize==8
        else:
            print(f'boxsize {boxsize} not supported yet')

        l = models.utils.LinearWN
        #c = models.utils.ConvTranspose2dWN
        c = models.utils.ConvTranspose2dWNUB
        v = models.utils.Conv2dWN
        #a = CenteredLeakyReLU
        a = nn.LeakyReLU
        s = nn.Sequential

        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        inch = 16 + 16 if not disable_id_encoder else 16
        if self.viewcond:
            self.viewmod = s(l( 3,    16), a(0.2, inplace=True),
                             l(16, 8*4*4), a(0.2, inplace=True))
            inch += 8

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize * self.outch]
            scale_factor = 4
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize * self.outch]
            scale_factor = 2
        else:
            print(f'Unsupported image size: {size}')
            quit()
        self.nlayers = len(size)-1

        h = 8
        for i in range(self.nlayers):
            #t = [c(size[i], size[i+1], 4, 2, 1)]
            t = [c(size[i], size[i+1], h, h, 4, 2, 1)]
            h *= 2

            if i < self.nlayers-1:
                t.append(a(0.2, inplace=True))
            self.add_module(f't{i}', s(*t))

        if self.texwarp:
            self.warpmod = s(v(inch, 256, 1, 1, 0), a(0.2, inplace=True),
                             c( 256, 256, 4, 2, 1), a(0.2, inplace=True),
                             c( 256, 128, 4, 2, 1), a(0.2, inplace=True),
                             c( 128, 128, 4, 2, 1), a(0.2, inplace=True),
                             c( 128,  64, 4, 2, 1), a(0.2, inplace=True),
                             c(  64,  64, 4, 2, 1), a(0.2, inplace=True),
                             c(  64,   2, 4, 2, 1),
                             nn.Upsample(scale_factor=scale_factor, mode='bilinear'))

        #self.apply(lambda x: he_init(x, 0.2))
        #he_init(self._modules[f't{self.nlayers-1}'][-1], 1)
        #if self.texwarp:
        #    he_init(self.warpmod[-2], 1)

        if self.viewcond:
            models.utils.initseq(self.viewmod)
        models.utils.initseq(self.encmod)
        for i in range(self.nlayers):
            models.utils.initseq(self._modules[f't{i}'])
        if self.texwarp:
            models.utils.initseq(self.warpmod)





        self.bias = nn.Parameter(torch.zeros(self.boxsize * self.outch, imsize, imsize))
        self.bias.data.zero_()

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, imsize),
                                   np.linspace(-1.0, 1.0, imsize))
        grid = np.concatenate((xgrid[None, :, :],
                               ygrid[None, :, :]),
                              axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpbias", torch.from_numpy(grid))


    def forward(self, ex_enc, id_enc, id_gainbias, view = None, use_warp = True, iternum = -1):

        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        if self.viewcond:
            v = self.viewmod(view).view(-1, 8, 4, 4)
            x = torch.cat([v, x], dim=1)
        x_orig = x

        ###############################################################################################################################
        scale = 1 / sqrt(2)
        for i in range(self.nlayers):
            xx = self._modules[f't{i}'](x)

            if id_gainbias is not None:
                n = id_gainbias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_gainbias[i][:,:n,...]*0.01 + 1.0) + id_gainbias[i][:,n:,...]) * scale
                elif n*2 == xx.shape[1]:
                    x = (xx + id_gainbias[i]) * scale
                else:
                    x = xx #note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

        # #test without skip connections
        # for i in range(self.nlayers):
        #     x = self._modules[f't{i}'](x)
        ###############################################################################################################################

        if self.texwarp and use_warp:
            w = self.warpmod(x_orig)
            w = w / self.imsize + self.warpbias
            x = torch.nn.functional.grid_sample(x, w.permute(0, 2, 3, 1))
        else:
            w = None
        tex = x + self.bias[None, :, :, :]

        x0 = tex
        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)
        x = x.view(x.size(0), self.boxsize, self.outch, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        x = x.reshape(x.size(0), self.nboxes, self.outch, self.boxsize, self.boxsize, self.boxsize)

        return x


###############################################################################
#add per-pixel gains as well as bias
#add geometry and motion as output, remove warp option, remove viewcond option
#positivity for alpha
class DecoderGeoSlab2(nn.Module):
    def __init__(self, uv, tri, uvtri, nvtx, motion_size, geo_size, imsize, nboxes, boxsize, disable_id_encoder=False):
        super(DecoderGeoSlab2, self).__init__()


        assert(motion_size < imsize)
        assert(geo_size < imsize)

        self.motion_size = motion_size
        self.geo_size = geo_size
        self.imsize = imsize
        self.nboxes = nboxes
        self.boxsize = boxsize


        nh = int(np.sqrt(self.nboxes))
        assert nh*nh==self.nboxes
        if nh==512:
            assert boxsize==2
        elif nh==64:
            assert boxsize==16
        elif nh==128:
            assert boxsize==8
        else:
            print(f'boxsize {boxsize} not supported yet')

        l = models.utils.LinearWN
        c = models.utils.ConvTranspose2dWNUB
        v = models.utils.Conv2dWN
        a = nn.LeakyReLU
        s = nn.Sequential

        #reduce noise effect of latent expression code
        self.encmod = s(v(16, 16, 1, 1, 0), a(0.2, inplace=True))
        models.utils.initseq(self.encmod)

        inch = 16 + 16 if not disable_id_encoder else 16 #first is for expression, second for identity

        if imsize == 1024:
            size = [inch, 256, 128, 128, 64, 64, 32, 16, self.boxsize]
            scale_factor = 4
        elif imsize == 512:
            size = [inch, 256, 128, 128, 64, 64, 32, self.boxsize]
            scale_factor = 2
        else:
            print(f'Unsupported image size: {size}')
            quit()
        self.nlayers = len(size)-1

        #build deconv arch with early exists for geometry and motion
        h = 8
        for i in range(self.nlayers):
            t = [c(size[i], size[i+1], h, h, 4, 2, 1)]
            if i < self.nlayers-1:
                t.append(a(0.2, inplace=True))
            self.add_module(f't{i}', s(*t))
            models.utils.initseq(self._modules[f't{i}'])

            if h == motion_size:
                self.motion = s(v(size[i+1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 9, 1, 1, 0))
                models.utils.initseq(self.motion)

            if h == geo_size:
                self.geo = s(v(size[i+1], 64, 1, 1, 0), a(0.2, inplace=True), v(64, 3, 1, 1, 0))
                models.utils.initseq(self.geo)

            h *= 2


        self.bias = nn.Parameter(torch.zeros(self.boxsize, imsize, imsize))
        self.bias.data.zero_()

        xgrid, ygrid = np.meshgrid(np.linspace(-1.0, 1.0, imsize),
                                   np.linspace(-1.0, 1.0, imsize))
        grid = np.concatenate((xgrid[None, :, :],
                               ygrid[None, :, :]),
                              axis=0)[None, ...].astype(np.float32)
        self.register_buffer("warpbias", torch.from_numpy(grid))



        #create cropping coordinates for geometry points
        vlists = [list() for _ in range(nvtx)]

        print(f"{nvtx=}")
        print(f"{tri.shape=}")
        print(f"{uvtri.shape=}")

        try:
            for fi in range(tri.shape[0]):
                for fv in range(3):
                    vlists[tri[fi,fv]].append(uvtri[fi,fv])
        except IndexError:
            print(f"{fi=}")
            print(f"{fv=}")
            print(f"{tri[fi,fv]=}")
            print(f"{uvtri[fi,fv]=}")
            raise
        nMaxUVsPerVertex = np.max([len(v) for v in vlists])
        print('Max UVs per vertex: {}'.format(nMaxUVsPerVertex))
        nMaxUVsPerVertex = 1#2
        uvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.int32)
        wuvspervert = np.zeros((nvtx, nMaxUVsPerVertex), dtype=np.float32)
        uvmask = np.ones((nvtx,), dtype=np.float32)
        for tvi in range(len(vlists)):
            if not (len(vlists[tvi])):
                uvmask[tvi] = 0
                continue
            for vsi in range(nMaxUVsPerVertex):
                if vsi<len(vlists[tvi]):
                    uvspervert[tvi,vsi] = vlists[tvi][vsi]
                    wuvspervert[tvi,vsi] = 1.0/nMaxUVsPerVertex
                elif len(vlists[tvi]):
                    uvspervert[tvi,vsi] = vlists[tvi][0]
                    wuvspervert[tvi,vsi] = 1.0/nMaxUVsPerVertex
        self.register_buffer("t_nl_uvspervert",
                             torch.from_numpy(uvspervert).long().to("cuda"))
        self.register_buffer("t_nl_wuvspervert",
                             torch.from_numpy(wuvspervert).to("cuda"))
        t_nl_geom_vert_uvs = torch.from_numpy(uv).to("cuda")[self.t_nl_uvspervert,:]
        coords = t_nl_geom_vert_uvs.view(1, -1, nMaxUVsPerVertex, 2) * 2 - 1.0
        self.register_buffer("coords", coords)


    def forward(self, ex_enc, id_enc, id_gainbias, iternum = -1):

        z = self.encmod(ex_enc).view(-1, 16, 4, 4)
        x = torch.cat([z, id_enc], dim=1) if id_enc is not None else z

        scale = 1 / sqrt(2)
        for i in range(self.nlayers):
            xx = self._modules[f't{i}'](x)

            if id_gainbias is not None:
                n = id_gainbias[i].shape[1] // 2
                if n == xx.shape[1]:
                    x = (xx * (id_gainbias[i][:,:n,...]*0.1 + 1.0) + id_gainbias[i][:,n:,...]) * scale
                elif n*2 == xx.shape[1]:
                    x = (xx + id_gainbias[i]) * scale
                else:
                    x = xx #note: last layer (1024x1024) ignores the pass through since slab is larger than 3 channels
            else:
                x = xx

            if x.shape[-1] == self.motion_size:
                mot = self.motion(x)
            if x.shape[-1] == self.geo_size:
                geo = self.geo(x)

        tex = torch.exp((x + self.bias[None, :, :, :]) * 0.1)


        #get motion
        mot = mot.view(mot.size(0), 9, -1).permute(0, 2, 1).contiguous()
        primposresid = mot[:, :, 0:3] * 0.01
        primrvecresid = mot[:, :, 3:6] * 0.01
        primscaleresid = torch.exp(0.01 * mot[:, :, 6:9])

        #get geometry
        coords = self.coords.expand((geo.size(0), -1, -1, -1))
        geo = F.grid_sample(geo, coords).mean(dim=3).permute(0, 2, 1)


        x0 = tex
        x = tex
        h = int(np.sqrt(self.nboxes))
        w = int(h)
        x = x.view(x.size(0), self.boxsize, 1, h, self.boxsize, w, self.boxsize)
        x = x.permute(0, 3, 5, 2, 1, 4, 6)

        x = x.reshape(x.size(0), self.nboxes, 1, self.boxsize, self.boxsize, self.boxsize)


        return x, geo, primposresid, primrvecresid, primscaleresid
