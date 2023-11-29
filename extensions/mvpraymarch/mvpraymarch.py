from typing import Dict, Optional, Tuple

import torch as th

from extensions.mvpraymarch import mvpraymarch_ext as _mvpraymarch_ext

th.ops.load_library(_mvpraymarch_ext.__file__)

def build_accel2(primpos, primrot, primscale, fixedorder: bool=False,
        fixed_bvh_cache : Optional[Dict[int, Tuple[th.Tensor, th.Tensor, th.Tensor]]] = None):
    """build bvh structure given primitive centers and sizes
    
    Parameters:
    ----------
    centers: N x K x 3 tensor of primitive centers
    invsig: N x K tensor of primitive inverse radius
    fixedorder: True means the bvh builder will not reorder primitives and will
    use a trivial tree structure. Likely to be slow for arbitrary
    configurations of primitives.
    
    """
    N = primpos.size(0)
    K = primpos.size(1)
    dev = primpos.device


    if fixedorder:
        # idk just picked this randomly, TorchScript doesn't upport tuple keys.
        assert N < 64
        key = (K << 6) | N

        if fixed_bvh_cache is not None and key in fixed_bvh_cache:
            nodechildren, nodeparent, sortedobjid = fixed_bvh_cache[key]
            nodechildren = nodechildren.to(dev)
            nodeparent = nodeparent.to(dev)
            sortedobjid = sortedobjid.to(dev)
        else:
            nodechildren = th.cat([
                (th.arange((K - 1) * 2, dtype=th.int32, device=dev) + 1).view(K - 1, 2),
                th.stack([
                    -(th.arange(K, dtype=th.int32, device=dev) + 1),
                    -(th.arange(K, dtype=th.int32, device=dev) + 2)], dim=-1)], dim=0)\
                            [None, :, :].repeat(N, 1, 1)
            nodeparent = th.stack([
                (th.arange(K, dtype=th.int32, device=dev) - 1),
                (th.arange(K, dtype=th.int32, device=dev) - 1)], dim=-1).view(-1)\
                        [None, 1:].repeat(N, 1)
            sortedobjid = th.arange(K, dtype=th.int32, device=dev)[None].repeat(N, 1)

            if fixed_bvh_cache is not None:
                fixed_bvh_cache[key] = (nodechildren, nodeparent, sortedobjid)
    else:
        # compute and sort morton codes
        cmax = primpos.max(dim=1, keepdim=True)[0]
        cmin = primpos.min(dim=1, keepdim=True)[0]
        centers_norm = (primpos - cmin) / (cmax - cmin).clamp(min=1e-8)
        centerinvsig_norm = th.cat((centers_norm, 0. * centers_norm[:, :, 0:1]), dim=2)

        objid = th.arange(K, dtype=th.int32, device=dev)[None].repeat(N, 1)
        sortedobjid = th.empty((N, K), dtype=th.int32, device=dev)
        mortoncode = th.empty((N, K), dtype=th.int32, device=dev)
        sortedcode = th.empty((N, K), dtype=th.int32, device=dev)

        th.ops.mvpraymarch_ext.compute_morton(centerinvsig_norm, mortoncode)
        sortedcode, sortedobjid_long = th.sort(mortoncode, dim=-1)
        sortedobjid = sortedobjid_long.int()

        nodechildren = th.empty((N, K + K - 1, 2), dtype=th.int32, device=dev)
        nodeparent = -th.ones((N, K + K - 1), dtype=th.int32, device=dev)
        th.ops.mvpraymarch_ext.build_tree(sortedcode, nodechildren, nodeparent)

    nodeaabb = th.empty((N, K + K - 1, 2, 3), dtype=th.float32, device=dev)
    th.ops.mvpraymarch_ext.compute_aabb2(sortedobjid, primpos, primrot, primscale, nodechildren, nodeparent, nodeaabb)

    return sortedobjid, nodechildren, nodeaabb

class MVPRaymarch(th.autograd.Function):
    """Custom Function for raymarching Mixture of Volumetric Primitives."""
    @staticmethod
    def forward(self, raypos, raydir, stepsize, tminmax, template, warp, primpos,
            primrot, primscale, gradmode, options,
            fixed_bvh_cache: Optional[Dict[int, Tuple[th.Tensor, th.Tensor, th.Tensor]]]=None):
        usebvh = options["usebvh"]
        randomorder = options["randomorder"]
        chlast = options["chlast"]
        fadescale = options["fadescale"]
        fadeexp = options["fadeexp"]
        accum = options["accum"]
        assert accum in [0, 1, 2]
        termthresh = options["termthresh"]
        with_t_img = options["with_t_img"]
        if accum == 1:
            assert termthresh >= 0
        elif accum == 2:
            assert termthresh >= 1
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
            sortedobjid, nodechildren, nodeaabb = build_accel2(
                primpos,
                primrot,
                primscale,
                fixedorder=usebvh=="fixedorder",
                fixed_bvh_cache=fixed_bvh_cache
            )
            assert sortedobjid.is_contiguous()
            assert nodechildren.is_contiguous()
            assert nodeaabb.is_contiguous()

            if randomorder:
                sortedobjid = sortedobjid[th.randperm(len(sortedobjid))]
        else:
            _, sortedobjid, nodechildren, nodeaabb = None, None, None, None

        # March through boxes
        rayrgba = th.empty((raypos.size(0), raypos.size(1), raypos.size(2), 4), device=raypos.device, dtype=template.dtype)

        raysat = None
        rayterm = None
        if gradmode:
            if accum == 0:
                raysat = th.full((raypos.size(0), raypos.size(1), raypos.size(2), 3), -1, device=raypos.device, dtype=th.float32)
            if accum > 0:
                rayterm = th.empty((raypos.size(0), raypos.size(1), raypos.size(2), 2), device=raypos.device, dtype=th.int32)

        t_img = None
        if with_t_img:
            t_img = th.full(
                (raypos.size(0), raypos.size(1), raypos.size(2)),
                th.inf,
                device=raypos.device, dtype=th.float32
            )

        th.ops.mvpraymarch_ext.raymarch_forward(raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb, template, warp,
                primpos, primrot, primscale, rayrgba, raysat, rayterm, t_img,
                chlast, fadescale, fadeexp, accum, termthresh,
                griddim, blocksizex, blocksizey)

        if not (
            template.requires_grad or
            (warp is not None and warp.requires_grad) or
            primpos.requires_grad or
            primrot.requires_grad or
            primscale.requires_grad
        ):
            return rayrgba, t_img

        self.save_for_backward(raypos, raydir, tminmax, sortedobjid,
                nodechildren, nodeaabb, template, warp, primpos, primrot, primscale,
                rayrgba, raysat, rayterm)

        self.options = options
        self.stepsize = stepsize

        return rayrgba, t_img

    @staticmethod
    def backward(self, grad_rayrgba, grad_t_img):
        raypos, raydir, tminmax, sortedobjid, nodechildren, nodeaabb, template, \
            warp, primpos, primrot, primscale, rayrgba, raysat, rayterm = self.saved_tensors
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

        th.ops.mvpraymarch_ext.raymarch_backward(raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb, template, warp, primpos,
                primrot, primscale, rayrgba, raysat, rayterm, grad_rayrgba.contiguous(),
                grad_template, grad_warp, grad_primpos, grad_primrot,
                grad_primscale, chlast, fadescale, fadeexp, accum, termthresh,
                griddim, blocksizex, blocksizey)

        return None, None, None, None, grad_template, grad_warp, grad_primpos, grad_primrot, grad_primscale, None, None

def mvpraymarch(raypos, raydir, stepsize: float, tminmax, template, warp: Optional[th.Tensor], primpos,
        primrot, primscale, usebvh: str = "fixedorder", randomorder: bool = False,
        chlast: bool = False, fadescale: float = 8., fadeexp: float = 8.,
        accum: int = 0, termthresh: float = 0., griddim: int = 3, blocksize: Tuple[int, int] = (6, 32),
        with_t_img: bool=False, fixed_bvh_cache: Optional[Dict[int, Tuple[th.Tensor, th.Tensor, th.Tensor]]] = None):
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
    griddim: Dimensionality of CUDA launch grid, can be 1, 2, or 3.
    blocksize: blocksize of CUDA kernels. Should be 2-element tuple if
        griddim=2 or griddim=3, or integer if griddim=1.
    with_t_img: return an additional image-shaped tensor containing accumulated
        ray t value weighted by opacity.
    """
    if accum == 2:
        termthresh = 1 / (1 - termthresh) # reparameterize so it has similar meaning when accum=1
    if th.jit.is_scripting():
        if isinstance(blocksize, tuple):
            blocksizex, blocksizey = blocksize
        else:
            blocksizex = blocksize
            blocksizey = 1

        # TODO: temp hack for scripting
        if usebvh != "fixedorder":
            assert usebvh in ["True", "False"]
            _usebvh = usebvh == "True"
        else:
            _usebvh = True

        # Build bvh
        if _usebvh:
            sortedobjid, nodechildren, nodeaabb = build_accel2(
                primpos,
                primrot,
                primscale,
                fixedorder=usebvh=="fixedorder",
                fixed_bvh_cache=fixed_bvh_cache
            )
        else:
            sortedobjid, nodechildren, nodeaabb = None, None, None

        # March through boxes
        rayrgba = th.empty(
            (raypos.size(0), raypos.size(1), raypos.size(2), 4),
            device=raypos.device, dtype=template.dtype
        )

        t_img: Optional[th.Tensor] = None
        if with_t_img:
            t_img = th.empty(
                (raypos.size(0), raypos.size(1), raypos.size(2)),
                device=raypos.device, dtype=th.float32
            )

        th.ops.mvpraymarch_ext.raymarch_forward(raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb, template, warp,
                primpos, primrot, primscale, rayrgba, None, None, t_img,
                chlast, fadescale, fadeexp, accum, termthresh,
                griddim, blocksizex, blocksizey)
    else:
        # TODO: temp hack for scripting
        if usebvh != "fixedorder":
            assert isinstance(usebvh, bool)

        rayrgba, t_img = MVPRaymarch.apply(raypos, raydir, stepsize, tminmax, template, warp,
                primpos, primrot, primscale, th.is_grad_enabled(),
                {"usebvh": usebvh, "randomorder": randomorder,
                    "chlast": chlast, "fadescale": fadescale, "fadeexp": fadeexp,
                    "accum": accum, "termthresh": termthresh,
                    "griddim": griddim, "blocksize": blocksize, "with_t_img": with_t_img})

    # Since we can't backprop through uint8 templates, and we want to preserve
    # the dtype of the output image (which wouldn't work for uint8 since we
    # couldn't output negative values for log-alpha), we do the exp'ing in the
    # CUDA code for uint8.
    if template.dtype != th.uint8:
        if accum == 1:
            # convert log alpha to alpha
            rayrgba = th.cat([rayrgba[..., :3], 1 - th.exp(-rayrgba[..., 3:4])], dim=-1)
        elif accum == 2:
            rayrgba = th.cat([
                rayrgba[..., :3], (termthresh * (1 - th.exp(-rayrgba[..., 3:4] / termthresh))).clamp(max=1)
            ], dim=-1)

    if with_t_img:
        assert t_img is not None
        alpha = rayrgba[..., 3]
        if template.dtype == th.uint8:
            alpha = alpha.float() / 255
        t_img /= alpha

    return rayrgba, t_img
