from typing import List, Dict, Tuple, Optional

import numpy as np
import time

import torch as th
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

try:
    from . import computeraydirs_ext as _computeraydirs_ext
except:
    import computeraydirs_ext as _computeraydirs_ext

th.ops.load_library(_computeraydirs_ext.__file__)

class ComputeRaydirs(Function):
    @staticmethod
    def forward(self, viewpos, viewrot, focal, princpt, pixelcoords: Optional[th.Tensor], volradius: float, size: Optional[Tuple[int, int]]=None):
        for tensor in [viewpos, viewrot, focal, princpt]:
            assert tensor.is_contiguous()

        N = viewpos.shape[0]
        if size is not None:
            H, W = size
        else:
            assert pixelcoords is not None
            assert pixelcoords.is_contiguous()
            H = pixelcoords.size(1)
            W = pixelcoords.size(2)

        raypos = th.empty((N, H, W, 3), device=viewpos.device)
        raydirs = th.empty((N, H, W, 3), device=viewpos.device)
        tminmax = th.empty((N, H, W, 2), device=viewpos.device)
        th.ops.computeraydirs_ext.compute_raydirs_forward(viewpos, viewrot,
                focal, princpt, pixelcoords, H, W, volradius, raypos, raydirs,
                tminmax)

        return raypos, raydirs, tminmax

    @staticmethod
    def backward(self, grad_raypos, grad_raydirs, grad_tminmax):
        # TODO:

        return None, None, None, None, None, None, None

def compute_raydirs(viewpos, viewrot, focal, princpt, pixelcoords: Optional[th.Tensor], volradius: float, size: Optional[Tuple[int, int]]=None):
    if th.jit.is_scripting():
        for tensor in [viewpos, viewrot, focal, princpt]:
            assert tensor.is_contiguous()

        N = viewpos.shape[0]
        if size is not None:
            H, W = size
        else:
            assert pixelcoords is not None
            assert pixelcoords.is_contiguous()
            H = pixelcoords.size(1)
            W = pixelcoords.size(2)

        raypos = th.empty((N, H, W, 3), device=viewpos.device)
        raydirs = th.empty((N, H, W, 3), device=viewpos.device)
        tminmax = th.empty((N, H, W, 2), device=viewpos.device)
        th.ops.computeraydirs_ext.compute_raydirs_forward(viewpos, viewrot,
                focal, princpt, pixelcoords, H, W, volradius, raypos, raydirs,
                tminmax)
    else:
        raypos, raydirs, tminmax = ComputeRaydirs.apply(viewpos, viewrot, focal, princpt, pixelcoords, volradius, size)

    return raypos, raydirs, tminmax
