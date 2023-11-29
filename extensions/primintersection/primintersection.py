import torch as th

from extensions.primintersection import primintersection_ext
from extensions.mvpraymarch.mvpraymarch import build_accel2

th.ops.load_library(primintersection_ext.__file__)


class PrimIntersection(th.autograd.Function):
    """Custom Function for raymarching Mixture of Volumetric Primitives."""

    @staticmethod
    def forward(self, raypos, raydir, primpos, primrot, primscale, options):
        usebvh = options["usebvh"]
        griddim = options["griddim"]
        if isinstance(options["blocksize"], tuple):
            blocksizex, blocksizey = options["blocksize"]
        else:
            blocksizex = options["blocksize"]
            blocksizey = 1

        assert raypos.is_contiguous() and raypos.size(3) == 3
        assert raydir.is_contiguous() and raydir.size(3) == 3
        assert primpos.is_contiguous() and primpos.size(2) == 3
        assert primrot.is_contiguous() and primrot.size(2) == 3
        assert primscale.is_contiguous() and primscale.size(2) == 3

        # you get weird artifacts if the primitives are really small, so
        primscale = th.where(
            primscale.max(dim=-1, keepdim=True)[0] > 1024.0,
            primscale * 1024.0 / primscale.max(dim=-1, keepdim=True)[0],
            primscale,
        )

        # Build bvh
        if usebvh is not False:
            sortedobjid, nodechildren, nodeaabb = build_accel2(
                primpos, primrot, primscale, fixedorder=usebvh == "fixedorder"
            )
            assert sortedobjid.is_contiguous()
            assert nodechildren.is_contiguous()
            assert nodeaabb.is_contiguous()
        else:
            assert False
            _, sortedobjid, nodechildren, nodeaabb = None, None, None, None

        # March through boxes
        raysteps = th.empty(
            (raypos.size(0), raypos.size(1), raypos.size(2), 1), device=raypos.device
        )
        th.ops.primintersection_ext.primintersection_forward(
            raypos,
            raydir,
            sortedobjid,
            nodechildren,
            nodeaabb,
            primpos,
            primrot,
            primscale,
            raysteps,
            griddim,
            blocksizex,
            blocksizey,
        )

        if not (
            primpos.requires_grad or
            primrot.requires_grad or
            primscale.requires_grad
        ):
            return raysteps

        self.save_for_backward(
            raypos,
            raydir,
            sortedobjid,
            nodechildren,
            nodeaabb,
            primpos,
            primrot,
            primscale,
            raysteps,
        )
        self.options = options

        return raysteps

    @staticmethod
    def backward(self, grad_raysteps):
        (
            raypos,
            raydir,
            sortedobjid,
            nodechildren,
            nodeaabb,
            primpos,
            primrot,
            primscale,
            raysteps,
        ) = self.saved_tensors
        griddim = self.options["griddim"]
        if isinstance(self.options["blocksize"], tuple):
            blocksizex, blocksizey = self.options["blocksize"]
        else:
            blocksizex = self.options["blocksize"]
            blocksizey = 1

        grad_primpos = th.zeros_like(primpos)
        grad_primrot = th.zeros_like(primrot)
        grad_primscale = th.zeros_like(primscale)

        th.ops.primintersection_ext.primintersection_backward(
            raypos,
            raydir,
            sortedobjid,
            nodechildren,
            nodeaabb,
            primpos,
            primrot,
            primscale,
            raysteps,
            grad_raysteps.contiguous(),
            grad_primpos,
            grad_primrot,
            grad_primscale,
            griddim,
            blocksizex,
            blocksizey,
        )

        return None, None, grad_primpos, grad_primrot, grad_primscale, None


def primintersection(
    raypos,
    raydir,
    primpos,
    primrot,
    primscale,
    usebvh="fixedorder",
    griddim=3,
    blocksize=(8, 16),
):
    """Main entry point for primintersection.

    Parameters:
    ----------
    raypos: N x H x W x 3 tensor of ray origins
    raydir: N x H x W x 3 tensor of ray directions
    primpos: N x K x 3 tensor of primitive centers
    primrot: N x K x 3 x 3 tensor of primitive orientations
    primscale: N x K x 3 tensor of primitive inverse dimension lengths
    usebvh: True to use bvh, "fixedorder" for a simple BVH, False for no bvh
    griddim: CUDA launch dimensionality
    blocksize: blocksize of CUDA kernels. Should be 2-element tuple if
        griddim=2 or 3, or integer if griddim=1."""
    out = PrimIntersection.apply(
        raypos,
        raydir,
        primpos,
        primrot,
        primscale,
        {"usebvh": usebvh, "griddim": griddim, "blocksize": blocksize},
    )
    return out
