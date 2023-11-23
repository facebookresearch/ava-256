import math
import operator
from collections import OrderedDict
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


### normal initialization routines
def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, "bias"):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

    if (
        isinstance(m, Conv2dWNUB)
        or isinstance(m, Conv2dWN)
        or isinstance(m, ConvTranspose2dWN)
        or isinstance(m, ConvTranspose2dWNUB)
        or isinstance(m, LinearWN)
    ):
        norm = np.sqrt(torch.sum(m.weight.data[:] ** 2))
        m.g.data[:] = norm


def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain("relu"))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain("leaky_relu", b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])


### custom modules
class LinearWN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWN, self).__init__(in_features, out_features, bias)
        self.g = nn.Parameter(torch.ones(out_features))
        self.fused = False

    def fuse(self):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        self.weight.data = self.weight.data * self.g.data[:, None] / wnorm
        self.fused = True

    def forward(self, input):
        if self.fused:
            return F.linear(input, self.weight, self.bias)
        else:
            wnorm = torch.sqrt(torch.sum(self.weight**2))
            return F.linear(input, self.weight * self.g[:, None] / wnorm, self.bias)


class Downsample2d(nn.Module):
    def __init__(self, nchannels, stride=1, padding=0):
        super(Downsample2d, self).__init__()

        self.nchannels = nchannels
        self.stride = stride
        self.padding = padding

        blurkernel = torch.tensor([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        blurkernel = blurkernel[:, None] * blurkernel[None, :]
        blurkernel = blurkernel / torch.sum(blurkernel)
        blurkernel = blurkernel[None, None, :, :].repeat(nchannels, 1, 1, 1)
        self.register_buffer("kernel", blurkernel)

    def forward(self, x):
        if self.padding == "reflect":
            x = F.pad(x, (3, 3, 3, 3), mode="reflect")
            return F.conv2d(x, weight=self.kernel, stride=self.stride, padding=0, groups=self.nchannels)
        else:
            return F.conv2d(x, weight=self.kernel, stride=self.stride, padding=self.padding, groups=self.nchannels)


class Dilate2d(nn.Module):
    def __init__(self, nchannels, kernelsize, stride=1, padding=0):
        super(Dilate2d, self).__init__()

        self.nchannels = nchannels
        self.kernelsize = kernelsize
        self.stride = stride
        self.padding = padding

        # blurkernel = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        # blurkernel = torch.tensor([1., 1., 1.])
        blurkernel = torch.ones((self.kernelsize,))
        blurkernel = blurkernel[:, None] * blurkernel[None, :]
        blurkernel = blurkernel / torch.sum(blurkernel)
        blurkernel = blurkernel[None, None, :, :].repeat(nchannels, 1, 1, 1)
        self.register_buffer("kernel", blurkernel)

    def forward(self, x):
        return F.conv2d(x, weight=self.kernel, stride=self.stride, padding=self.padding, groups=self.nchannels).clamp(
            max=1.0
        )


class CoordConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CoordConv2d, self).__init__(
            in_channels + 2, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        x = torch.cat(
            [
                x,
                torch.linspace(-1.0, 1.0, x.size(2), device=x.device)[None, None, :, None].repeat(
                    x.size(0), 1, 1, x.size(3)
                ),
                torch.linspace(-1.0, 1.0, x.size(3), device=x.device)[None, None, None, :].repeat(
                    x.size(0), 1, x.size(2), 1
                ),
            ],
            dim=1,
        )
        return F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2dWS(nn.Conv2d):
    """Weight standardization from NFNets"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWS, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weights(self, eps):
        mean = torch.mean(self.weight, dim=(1, 2, 3), keepdims=True)
        var = torch.var(self.weight, dim=(1, 2, 3), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[1:]))

        scale = (
            1.414
            * torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps).to(var.device)))
            * self.gain.view_as(var).to(var.device)
        )
        # shift = mean * scale
        # return self.weight * scale - shift
        return (self.weight - mean) * scale

    def forward(self, x):
        weight = self.standardize_weights(1e-4)
        out = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print("!", out.mean().item(), out.std().item())
        return out


class Conv2dWN(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True)
        self.g = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return F.conv2d(
            x,
            self.weight * self.g[:, None, None, None] / wnorm,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2dUB(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dUB, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        return (
            F.conv2d(
                x,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class Conv2dWNUB(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2dWNUB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        return (
            F.conv2d(
                x,
                self.weight * self.g[:, None, None, None] / wnorm,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class ConvTranspose2dWN(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvTranspose2dWN, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, True
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.fused = False

    def fuse(self):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        self.weight.data = self.weight.data * self.g.data[None, :, None, None] / wnorm
        self.fused = True

    def forward(self, x):
        bias = self.bias
        assert bias is not None
        if self.fused:
            return F.conv_transpose2d(
                x,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            wnorm = torch.sqrt(torch.sum(self.weight**2))
            return F.conv_transpose2d(
                x,
                self.weight * self.g[None, :, None, None] / wnorm,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )


class ConvTranspose2dUB(nn.ConvTranspose2d):
    def __init__(
        self,
        width,
        height,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ConvTranspose2dUB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        return (
            F.conv_transpose2d(
                x,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class ConvTranspose2dWNUB(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ConvTranspose2dWNUB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False
        )
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))
        # self.biasf = nn.Parameter(torch.zeros(out_channels, height, width))
        self.fused = False

    def fuse(self):
        wnorm = torch.sqrt(torch.sum(self.weight**2))
        self.weight.data = self.weight.data * self.g.data[None, :, None, None] / wnorm
        self.fused = True

    def forward(self, x):
        bias = self.bias
        assert bias is not None
        if self.fused:
            return (
                F.conv_transpose2d(
                    x,
                    self.weight,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                + bias[None, ...]
            )
        else:
            wnorm = torch.sqrt(torch.sum(self.weight**2))
            return (
                F.conv_transpose2d(
                    x,
                    self.weight * self.g[None, :, None, None] / wnorm,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                + bias[None, ...]
            )


class Conv3dUB(nn.Conv3d):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv3dUB, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels, depth, height, width))

    def forward(self, x):
        return (
            F.conv3d(
                x,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class ConvTranspose3dWS(nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ConvTranspose3dWS, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weights(self, eps):
        mean = torch.mean(self.weight, dim=(1, 2, 3, 4), keepdims=True)
        var = torch.var(self.weight, dim=(1, 2, 3, 4), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[1:]))

        scale = (
            1.414
            * 2.0
            * torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps).to(var.device)))
            * self.gain.view_as(var).to(var.device)
        )
        # shift = mean * scale
        # return self.weight * scale - shift
        return (self.weight - mean) * scale

    def forward(self, x):
        weight = self.standardize_weights(1e-4)
        out = F.conv_transpose3d(
            x, weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
        # print("@", out.mean().item(), out.std().item())
        return out


class ConvTranspose3dUB(nn.ConvTranspose3d):
    def __init__(
        self,
        width,
        height,
        depth,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ConvTranspose3dUB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, depth, height, width))

    def forward(self, x):
        return (
            F.conv_transpose3d(
                x,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            + self.bias[None, ...]
        )


class Rodrigues(nn.Module):
    def __init__(self):
        super(Rodrigues, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec**2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack(
            (
                rvec[:, 0] ** 2 + (1.0 - rvec[:, 0] ** 2) * costh,
                rvec[:, 0] * rvec[:, 1] * (1.0 - costh) - rvec[:, 2] * sinth,
                rvec[:, 0] * rvec[:, 2] * (1.0 - costh) + rvec[:, 1] * sinth,
                rvec[:, 0] * rvec[:, 1] * (1.0 - costh) + rvec[:, 2] * sinth,
                rvec[:, 1] ** 2 + (1.0 - rvec[:, 1] ** 2) * costh,
                rvec[:, 1] * rvec[:, 2] * (1.0 - costh) - rvec[:, 0] * sinth,
                rvec[:, 0] * rvec[:, 2] * (1.0 - costh) - rvec[:, 1] * sinth,
                rvec[:, 1] * rvec[:, 2] * (1.0 - costh) + rvec[:, 0] * sinth,
                rvec[:, 2] ** 2 + (1.0 - rvec[:, 2] ** 2) * costh,
            ),
            dim=1,
        ).view(-1, 3, 3)


class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec**2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack(
            (
                1.0 - 2.0 * rvec[:, 1] ** 2 - 2.0 * rvec[:, 2] ** 2,
                2.0 * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
                1.0 - 2.0 * rvec[:, 0] ** 2 - 2.0 * rvec[:, 2] ** 2,
                2.0 * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
                2.0 * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
                1.0 - 2.0 * rvec[:, 0] ** 2 - 2.0 * rvec[:, 1] ** 2,
            ),
            dim=1,
        ).view(-1, 3, 3)


class BufferDict(nn.Module):
    def __init__(self, d, persistent=False):
        super(BufferDict, self).__init__()

        for k in d:
            self.register_buffer(k, d[k], persistent=False)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, parameter):
        self.register_buffer(key, parameter, persistent=False)


# TODO: remove?
class LightEnvmapDecorator(nn.Module):
    def __init__(self, mod, dataset, L, lightinitpath=None):
        super(LightEnvmapDecorator, self).__init__()

        self.mod = mod

        if lightinitpath is not None:
            lightinit = torch.from_numpy(np.log(0.0001 + np.load(lightinitpath)))
            self.light = nn.ParameterList([nn.Parameter(lightinit[i]) for i in range(len(dataset.get_light_ids()))])
        else:
            self.light = nn.ParameterList(
                [nn.Parameter(0.1 * torch.randn(1, L, 2 * L)) for i in range(len(dataset.get_light_ids()))]
            )

    def forward(self, lightid, **kwargs):
        light = torch.stack([self.light[lightid[i].item()] for i in range(lightid.size(0))], dim=0)
        return self.mod(light=light, **kwargs)


class LightEnvmapRotateDecorator(nn.Module):
    def __init__(self, mod, period=200.0):
        super(LightEnvmapRotateDecorator, self).__init__()

        self.mod = mod
        self.period = period

    def forward(self, lightid, **kwargs):
        light = []
        for i in range(campos.size(0)):
            light0 = -torch.ones((1, 4, 8))
            ph0 = (iternum * campos.size(0) + i) * 2.0 * 3.1415926 / self.period
            lightdir0 = torch.tensor([np.sin(ph0), 0.0, np.cos(ph0)])
            # lightdir0 = torch.tensor([0., np.sin(ph0), np.cos(ph0)])
            # lightdir0 = torch.tensor([np.sin(ph0), np.cos(ph0), 0.])
            for y in range(4):
                for x in range(8):
                    th = ((y + 0.5) / 4.0) * 3.1415926
                    ph = ((x + 0.5) / 4.0) * 3.1415926
                    dr = torch.tensor([np.sin(th) * np.sin(ph), np.cos(th), np.sin(th) * np.cos(ph)])
                    light0[:, y, x] = torch.log(
                        0.00001 + torch.exp(-4.0 * torch.acos(torch.dot(lightdir0, dr)) ** 2) * np.sin(th)
                    )
            light0 = light0 + 3.0 - torch.log(torch.sum(torch.exp(light0)))
            light.append(light0)
        light = torch.stack(light, dim=0).to("cuda")
        return self.mod(light=light, **kwargs)


def sh(m, n, ph, th):
    import scipy.special

    if m < 0:
        return (np.sqrt(2.0) * (-(1**m)) * np.imag(scipy.special.sph_harm(-m, n, ph, th))).astype(np.float32)
    elif m == 0:
        return (np.real(scipy.special.sph_harm(0, n, ph, th))).astype(np.float32)
    else:
        return (np.sqrt(2.0) * (-(1**m)) * np.real(scipy.special.sph_harm(m, n, ph, th))).astype(np.float32)


class SHLightDecorator(nn.Module):
    def __init__(self, mod, dataset, ncoeffs, lightinitpath=None):
        super(SHLightDecorator, self).__init__()

        self.mod = mod

        if lightinitpath is not None:
            # lightinit = torch.log(1e-3 + torch.from_numpy(np.load(lightinitpath)))
            lightinit = torch.from_numpy(np.load(lightinitpath))
            lightinit = torch.log(1e-3 + lightinit)

            norder = 5
            mlist = [x for m in range(norder) for x in range(-m, m + 1)]
            nlist = [m for m in range(norder) for x in range(-m, m + 1)]

            L = 16
            th, ph = np.meshgrid(
                (np.arange(L, dtype=np.float32) + 0.5) * 3.1415926 / L,
                (np.arange(2 * L, dtype=np.float32) + 0.5) * 3.1415926 / L,
                indexing="ij",
            )

            bases = torch.stack([torch.tensor(sh(m, n, ph, th)) for m, n in zip(mlist, nlist)], dim=0)  # [25, L, 2L]
            th, ph = torch.from_numpy(th), torch.from_numpy(ph)  # [L, 2L]

            # project onto SH
            coeffs = torch.stack(
                [
                    torch.stack(
                        [
                            torch.sum(lightinit[i, 0] * bases[j] * torch.sin(th[:, 0, None]) * np.pi**2 / (L * L))
                            for j in range(bases.size(0))
                        ],
                        dim=0,
                    )
                    for i in range(lightinit.size(0))
                ],
                dim=0,
            )

            # for i in range(lightinit.size(0)):
            #    print(i, coeffs[i].data.numpy())

            coeffs = coeffs[:, :, None].repeat(1, 1, 3).view(coeffs.size(0), -1)

            self.light = nn.ParameterList([nn.Parameter(coeffs[i]) for i in range(len(dataset.get_light_ids()))])
        else:
            self.light = nn.ParameterList(
                [nn.Parameter(0.1 * torch.randn(ncoeffs)) for i in range(len(dataset.get_light_ids()))]
            )

    def forward(self, iternum, lossweights, lightid, **kwargs):
        light = torch.stack([self.light[lightid[i].item()] for i in range(lightid.size(0))], dim=0)
        return self.mod(iternum, lossweights, light=light, **kwargs)


class SHLightRotateDecorator(nn.Module):
    def __init__(self, mod, period=200.0):
        super(SHLightRotateDecorator, self).__init__()

        self.mod = mod
        self.period = period

        norder = 5
        mlist = [x for m in range(norder) for x in range(-m, m + 1)]
        nlist = [m for m in range(norder) for x in range(-m, m + 1)]

        L = 16
        self.L = L
        th, ph = np.meshgrid(
            (np.arange(L, dtype=np.float32) + 0.5) * 3.1415926 / L,
            (np.arange(2 * L, dtype=np.float32) + 0.5) * 3.1415926 / L,
            indexing="ij",
        )

        self.bases = torch.stack([torch.tensor(sh(m, n, ph, th)) for m, n in zip(mlist, nlist)], dim=0)
        self.th, self.ph = torch.from_numpy(th), torch.from_numpy(ph)

    def forward(self, iternum, lossweights, lightid, **kwargs):
        light = []
        for i in range(lightid.size(0)):
            light0 = -torch.ones((1, self.L, self.L * 2))
            ph0 = (iternum * lightid.size(0) + i) * 2.0 * 3.1415926 / self.period
            lightdir0 = torch.tensor([np.sin(ph0), 0.0, np.cos(ph0)])
            # lightdir0 = torch.tensor([0., np.sin(ph0), np.cos(ph0)])
            # lightdir0 = torch.tensor([np.sin(ph0), np.cos(ph0), 0.])
            for y in range(self.L):
                for x in range(self.L * 2):
                    th = ((y + 0.5) / self.L) * 3.1415926
                    ph = ((x + 0.5) / self.L) * 3.1415926
                    dr = torch.tensor([np.sin(th) * np.sin(ph), np.cos(th), np.sin(th) * np.cos(ph)])
                    light0[:, y, x] = torch.log(
                        0.00001 + torch.exp(-4.0 * torch.acos(torch.dot(lightdir0, dr)) ** 2)
                    )  # * np.sin(th))
            light0 = light0 + 7.0 - torch.log(torch.sum(torch.exp(light0)))

            # project onto SH
            light0 = torch.stack(
                [
                    torch.sum(
                        (light0 + 1e-3) * self.bases[i] * torch.sin(self.th[:, 0, None]) * np.pi**2 / light0.numel()
                    )
                    for i in range(self.bases.size(0))
                ],
                dim=0,
            )

            light.append(light0)
        light = torch.stack(light, dim=0).to("cuda")
        light = light[:, :, None].repeat(1, 1, 3).view(light.size(0), -1)
        return self.mod(iternum, lossweights, light=light, **kwargs)


class SHNearfieldLightDecorator(nn.Module):
    def __init__(self, mod, period=131):
        super(SHNearfieldLightDecorator, self).__init__()

        self.mod = mod
        self.period = period

        norder = 5
        mlist = [x for m in range(norder) for x in range(-m, m + 1)]
        nlist = [m for m in range(norder) for x in range(-m, m + 1)]

        L = 8
        self.L = L
        th, ph = np.meshgrid(
            (np.arange(L, dtype=np.float32) + 0.5) * 3.1415926 / L,
            (np.arange(2 * L, dtype=np.float32) + 0.5) * 3.1415926 / L,
            indexing="ij",
        )
        th = th.ravel()
        ph = ph.ravel()

        self.bases = torch.stack([torch.tensor(sh(m, n, ph, th)) for m, n in zip(mlist, nlist)], dim=0)
        self.th, self.ph = torch.from_numpy(th), torch.from_numpy(ph)

    def forward(self, iternum, lossweights, lightid, **kwargs):
        # light = []
        # for i in range(lightid.size(0)):
        #    light0 = -torch.ones((1, self.L, self.L*2))
        #    ph0 = (iternum * lightid.size(0) + i) * 2. * 3.1415926 / self.period
        #    lightdir0 = torch.tensor([np.sin(ph0), 0., np.cos(ph0)])
        #    #lightdir0 = torch.tensor([0., np.sin(ph0), np.cos(ph0)])
        #    #lightdir0 = torch.tensor([np.sin(ph0), np.cos(ph0), 0.])
        #    for y in range(self.L):
        #        for x in range(self.L * 2):
        #            th = ((y+0.5) / self.L) * 3.1415926
        #            ph = ((x+0.5) / self.L) * 3.1415926
        #            dr = torch.tensor([np.sin(th) * np.sin(ph), np.cos(th), np.sin(th) * np.cos(ph)])
        #            light0[:, y, x] = torch.log(0.00001 + torch.exp(-4.0 * torch.acos(torch.dot(lightdir0, dr)) ** 2) * np.sin(th))
        #    light0 = light0 + 7. - torch.log(torch.sum(torch.exp(light0)))

        #    # project onto SH
        #    light0 = torch.stack([
        #        torch.sum((light0 + 1e-3) * self.bases[i] * torch.sin(self.th[:, 0, None]) * np.pi ** 2 / light0.numel())
        #        for i in range(self.bases.size(0))], dim=0)
        #
        #    light.append(light0)
        # light = torch.stack(light, dim=0).to("cuda")
        # light = light[:, :, None].repeat(1, 1, 3).view(light.size(0), -1)
        # return self.mod(iternum, lossweights, light=light, **kwargs)

        light = []
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 128), torch.linspace(-1.0, 1.0, 128), torch.linspace(-1.0, 1.0, 128)
        )
        # th, ph = torch.meshgrid(
        #        (torch.arange(self.L  ).float() + 0.5) * np.pi / self.L,
        #        (torch.arange(self.L*2).float() + 0.5) * np.pi / self.L)
        # th = th.contiguous().view(-1)
        # ph = ph.contiguous().view(-1)
        th, ph = self.th, self.ph
        vec = torch.stack(
            [torch.sin(th) * torch.sin(ph), torch.cos(th), torch.sin(th) * torch.cos(ph)], dim=-1
        )  # [2L^2, 3]
        xyz = torch.stack([xx, yy, zz], dim=-1)  # [D, D, D, 3]
        for i in range(lightid.size(0)):
            ph0 = (iternum * lightid.size(0) + i) * 2.0 * 3.1415926 / 131.0
            # lightpos = torch.tensor([-0.2, np.sin(ph0) * 0.3, 0.3])
            # lightpos = torch.tensor([-0.3, np.sin(ph0) * 0.3, 0.4])
            # lightpos = torch.tensor([-0.3, np.sin(ph0) * 0.3, 0.2]) # 3
            lightpos = torch.tensor([np.sin(ph0) * 0.3, 0.0, 0.4])  # 4
            # lightpos = torch.tensor([-0.3, 0.7, np.sin(ph0) * 0.3]) # 5
            # lightpos = torch.tensor([-0.2, 0.0, 0.3]) # test
            lightdir = lightpos[None, None, None, :] - xyz  # [D, D, D, 3]
            lightdirnorm = torch.sqrt(0.0001 + torch.sum(lightdir**2, dim=-1))  # [D, D, D]
            lightdirnormed = lightdir / lightdirnorm[:, :, :, None]
            dp = torch.sum(lightdirnormed[None, :, :, :, :] * vec[:, None, None, None, :], dim=-1)  # [2L^2, D, D, D]
            light0 = torch.exp(-16.0 * torch.acos(dp.clamp(min=-0.999, max=0.999)) ** 2) * torch.sin(
                th[:, None, None, None]
            )
            light0 = light0 / torch.sum(light0, dim=0, keepdim=True)
            light0 = light0 * 4.0 / lightdirnorm[None, :, :, :] ** 2
            # print(light0[:, 63, 63, 44].sum().item())
            light0 = torch.log(0.0001 + light0)

            # project onto SH
            # light0 = torch.stack([
            #    torch.sum(light0 * self.bases[i, :, None, None, None] * (np.pi ** 2 / (2 * self.L ** 2)), dim=0)
            #    for i in range(self.bases.size(0))], dim=0)
            light0 = torch.sum(light0[None, :, :, :, :] * self.bases[:, :, None, None, None], dim=1) * (
                np.pi**2 / (2 * self.L**2)
            )

            light.append(light0)
        light = torch.stack(light, dim=0).to("cuda")
        light = light[:, :, None, :, :, :].repeat(1, 1, 3, 1, 1, 1).view(light.size(0), -1, 128, 128, 128)
        return self.mod(iternum, lossweights, light=light, **kwargs)
