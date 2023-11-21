import torch as th

from extensions.integprior import integprior_ext
from extensions.mvpraymarch.mvpraymarch import build_accel2

th.ops.load_library(integprior_ext.__file__)


class IntegPrior(th.autograd.Function):
    """Custom Function for raymarching Mixture of Volumetric Primitives."""
    @staticmethod
    def forward(self, raypos, raydir, stepsize, tminmax, template, warp, primpos,
            primrot, primscale, options):
        usebvh = options["usebvh"]
        chlast = options["chlast"]
        fadescale = options["fadescale"]
        fadeexp = options["fadeexp"]
        accum = options["accum"]
        assert accum in [0, 1, 2]
        termthresh = options["termthresh"]
        if accum == 1:
            assert termthresh >= 0.
        elif accum == 2:
            assert termthresh >= 1.
        griddim = options["griddim"]
        if isinstance(options["blocksize"], tuple):
            blocksizex, blocksizey = options["blocksize"]
        else:
            blocksizex = options["blocksize"]
            blocksizey = 1

        assert raypos.is_contiguous() and raypos.size(3) == 3
        assert raydir.is_contiguous() and raydir.size(3) == 3
        assert tminmax.is_contiguous() and tminmax.size(3) == 2
        if chlast:
            assert template.is_contiguous() and len(template.size()) == 6 and template.size(-1) == 4
            assert warp is None or (warp.is_contiguous() and warp.size(-1) == 3)
        else:
            assert template.is_contiguous() and len(template.size()) == 6 and template.size(2) == 4
            assert warp is None or (warp.is_contiguous() and warp.size(2) == 3)
        assert primpos.is_contiguous() and primpos.size(2) == 3
        assert primrot.is_contiguous() and primrot.size(2) == 3
        assert primscale.is_contiguous() and primscale.size(2) == 3

        # Build bvh
        if usebvh is not False:
            # compute radius of primitives
            sortedobjid, nodechildren, nodeaabb = (
                build_accel2(primpos, primrot, primscale, fixedorder=usebvh == "fixedorder"))
            assert sortedobjid.is_contiguous()
            assert nodechildren.is_contiguous()
            assert nodeaabb.is_contiguous()
        else:
            _, sortedobjid, nodechildren, nodeaabb = None, None, None, None

        # March through boxes
        rayrgba = th.empty((raypos.size(0), raypos.size(1), raypos.size(2), 1), device=raypos.device)
        prior = th.empty((raypos.size(0), raypos.size(1), raypos.size(2), 1), device=raypos.device)
        raysat = -th.ones((raypos.size(0), raypos.size(1), raypos.size(2), 3), device=raypos.device)
        th.ops.integprior_ext.integprior_forward(raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb, template, warp,
                primpos, primrot, primscale, rayrgba, prior, raysat,
                chlast, fadescale, fadeexp, accum, termthresh,
                griddim, blocksizex, blocksizey)

        if not (
            template.requires_grad or
            (warp is not None and warp.requires_grad) or
            primpos.requires_grad or
            primrot.requires_grad or
            primscale.requires_grad
        ):
            return prior

        self.save_for_backward(raypos, raydir, tminmax, sortedobjid,
                nodechildren, nodeaabb, template, warp, primpos, primrot, primscale,
                rayrgba, raysat)
        self.options = options
        self.stepsize = stepsize

        return prior

    @staticmethod
    def backward(self, grad_prior):
        raypos, raydir, tminmax, sortedobjid, nodechildren, nodeaabb, template, \
            warp, primpos, primrot, primscale, rayrgba, raysat = self.saved_tensors
        chlast = self.options["chlast"]
        fadescale = self.options["fadescale"]
        fadeexp = self.options["fadeexp"]
        accum = self.options["accum"]
        termthresh = self.options["termthresh"]
        griddim = self.options["griddim"]
        if isinstance(self.options["blocksize"], tuple):
            blocksizex, blocksizey = self.options["blocksize"]
        else:
            blocksizex = self.options["blocksize"]
            blocksizey = 1

        stepsize = self.stepsize

        grad_template = th.zeros_like(template)
        grad_warp = th.zeros_like(warp) if warp is not None else None
        grad_primpos = th.zeros_like(primpos)
        grad_primrot = th.zeros_like(primrot)
        grad_primscale = th.zeros_like(primscale)

        th.ops.integprior_ext.integprior_backward(raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb, template, warp, primpos,
                primrot, primscale, rayrgba, raysat, grad_prior.contiguous(),
                grad_template, grad_warp, grad_primpos, grad_primrot,
                grad_primscale, chlast, fadescale, fadeexp, accum, termthresh,
                griddim, blocksizex, blocksizey)

        return None, None, None, None, grad_template, grad_warp, grad_primpos, grad_primrot, grad_primscale, None


def integprior(raypos, raydir, stepsize, tminmax, template, warp, primpos,
            primrot, primscale, usebvh="fixedorder",
            chlast=False, fadescale=8., fadeexp=8.,
            accum=0, termthresh=0., griddim=3, blocksize=(8, 16)):
    """Main entry point for raymarching MVP.

    Parameters:
    ----------
    raypos: N x H x W x 3 tensor of ray origins
    raydir: N x H x W x 3 tensor of ray directions
    stepsize: raymarching step size
    tminmax: N x H x W x 2 tensor of raymarching min/max bounds
    template: N x K x 4 x TD x TH x TW tensor of K RGBA primitives
    warp: N x K x 3 x TD x TH x TW tensor of K warp fields (optional)
    primpos: N x K x 3 tensor of primitive centers
    primrot: N x K x 3 x 3 tensor of primitive orientations
    primscale: N x K x 3 tensor of primitive inverse dimension lengths
    usebvh: True to use bvh, "fixedorder" for a simple BVH, False for no bvh
    chlast: whether template is provided as channels last or not. True tends
        to be faster.
    fadescale: Opacity is faded at the borders of the primitives by the equation
        exp(-fadescale * x ** fadeexp) where x is the normalized coordinates of
        the primitive.
    fadeexp: Opacity is faded at the borders of the primitives by the equation
        exp(-fadescale * x ** fadeexp) where x is the normalized coordinates of
        the primitive.
    accum : 0 to use additive raymarching accumulation (Neural Volumes
        style), 1 to use multiplicative (NeRF style), 2 to use hybrid.
    termthresh : for accum=1, raymarching terminates when alpha reaches this
        value. for accum=2, this controls whether multiplicative raymarching
        is used (termthresh=0), additive is used (termthresh=0.999), values
        in between have behavior which smoothly changes from additive to
        multiplicative. use a value like 0.1 to get multiplicative with early
        termination.
    griddim: CUDA launch dimensionality
    blocksize: blocksize of CUDA kernels. Should be 2-element tuple if
        griddim=2 or 3, or integer if griddim=1."""
    if accum == 2:
        termthresh = 1. / (1. - termthresh) # reparameterize so it has similar meaning when accum=1
    out = IntegPrior.apply(raypos, raydir, stepsize, tminmax, template, warp,
            primpos, primrot, primscale,
            {"usebvh": usebvh,
                "chlast": chlast, "fadescale": fadescale, "fadeexp": fadeexp,
                "accum": accum, "termthresh": termthresh,
                "griddim": griddim, "blocksize": blocksize})
    return out
