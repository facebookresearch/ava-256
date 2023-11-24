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
